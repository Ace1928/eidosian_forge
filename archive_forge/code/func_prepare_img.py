import argparse
import os
import align
import numpy as np
import requests
import tensorflow as tf
import torch
from PIL import Image
from tokenizer import Tokenizer
from transformers import (
from transformers.utils import logging
def prepare_img():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im