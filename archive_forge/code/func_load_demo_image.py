import argparse
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from transformers import (
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
def load_demo_image():
    url = 'https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    return image