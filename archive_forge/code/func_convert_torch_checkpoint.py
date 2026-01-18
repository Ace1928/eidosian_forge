import argparse
import re
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
def convert_torch_checkpoint(checkpoint, num_meta4D_last_stage):
    for key in checkpoint.copy().keys():
        val = checkpoint.pop(key)
        checkpoint[rename_key(key, num_meta4D_last_stage)] = val
    return checkpoint