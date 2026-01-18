import argparse
import os
import requests
import torch
from PIL import Image
from transformers import SuperPointConfig, SuperPointForKeypointDetection, SuperPointImageProcessor
def get_superpoint_config():
    config = SuperPointConfig(encoder_hidden_sizes=[64, 64, 128, 128], decoder_hidden_size=256, keypoint_decoder_dim=65, descriptor_decoder_dim=256, keypoint_threshold=0.005, max_keypoints=-1, nms_radius=4, border_removal_distance=4, initializer_range=0.02)
    return config