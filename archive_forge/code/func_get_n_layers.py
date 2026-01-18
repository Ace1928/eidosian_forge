import argparse
import json
import os.path
from collections import OrderedDict
import numpy as np
import requests
import torch
from flax.training.checkpoints import restore_checkpoint
from huggingface_hub import hf_hub_download
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
from transformers.image_utils import PILImageResampling
def get_n_layers(state_dict):
    return sum([1 if 'encoderblock_' in k else 0 for k in state_dict['optimizer']['target']['Transformer'].keys()])