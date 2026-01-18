import argparse
import json
import os
import numpy as np
import PIL
import requests
import tensorflow.keras.applications.efficientnet as efficientnet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing import image
from transformers import (
from transformers.utils import logging
def convert_image_processor(model_name):
    size = CONFIG_MAP[model_name]['image_size']
    preprocessor = EfficientNetImageProcessor(size={'height': size, 'width': size}, image_mean=[0.485, 0.456, 0.406], image_std=[0.47853944, 0.4732864, 0.47434163], do_center_crop=False)
    return preprocessor