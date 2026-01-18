import argparse
import os
import requests
import torch
from PIL import Image
from transformers import SuperPointConfig, SuperPointForKeypointDetection, SuperPointImageProcessor
def prepare_imgs():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im1 = Image.open(requests.get(url, stream=True).raw)
    url = 'http://images.cocodataset.org/test-stuff2017/000000004016.jpg'
    im2 = Image.open(requests.get(url, stream=True).raw)
    return [im1, im2]