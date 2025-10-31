# Real-Time Object Detection using YOLOv7

This project performs **real-time object detection** using **YOLOv7** and **OpenCV**.  
It captures video from your webcam and identifies multiple objects like people, cars, and animals instantly.

---

##  Features
- Real-time detection from webcam  
- Uses YOLOv7-Tiny for faster performance  
- Adjustable detection confidence and threshold values  
- Displays object labels and confidence scores live  

---

##  Requirements

Before running the project, make sure you have Python installed.  

```bash
pip install opencv-python numpy
```
## Project Structure
```
yolov-coco-v7
│
├── coco.names              # COCO dataset class labels
├── yolov7.cfg              # YOLOv7 configuration file
├── yolov7-tiny.weights     # Pre-trained YOLOv7-tiny model weights
└── realtime.py             # Main Python script for real-time detection
```
## How to Run

1. Clone this repository
```bash
git clone https://github.com/your-username/yolov7-realtime-detection.git
cd yolov7-realtime-detection
```
2. Activate your environment (if using conda)
conda activate yolov7-env
3. Run the script
```python
python realtime.py
```
4. Press Esc to exit.

