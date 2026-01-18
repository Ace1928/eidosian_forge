import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
@staticmethod
def load_video_frames(video_path, suffix_size):
    cap = cv2.VideoCapture(video_path)
    ret, frame_num = (True, 0)
    while ret:
        ret, _ = DataPipeline.read_frame(cap)
        if ret:
            frame_num += 1
    num_states = suffix_size
    frames = []
    max_frame_num = frame_num
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    for _ in range(max_frame_num - num_states):
        ret, _ = DataPipeline.read_frame(cap)
        frame_num += 1
        if not ret:
            return None
    while ret and frame_num < max_frame_num:
        ret, frame = DataPipeline.read_frame(cap)
        frames.append(frame)
        frame_num += 1
    return frames