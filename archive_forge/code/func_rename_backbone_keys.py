import argparse
from collections import OrderedDict
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import functional as F
from transformers import DetrImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection
from transformers.utils import logging
def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'backbone.0.body' in key:
            new_key = key.replace('backbone.0.body', 'backbone.conv_encoder.model')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict