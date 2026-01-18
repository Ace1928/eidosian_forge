import argparse
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T
from transformers import (
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def get_image():
    filepath = hf_hub_download(repo_id='hf-internal-testing/fixtures_docvqa', filename='document_2.png', repo_type='dataset')
    image = Image.open(filepath).convert('RGB')
    return image