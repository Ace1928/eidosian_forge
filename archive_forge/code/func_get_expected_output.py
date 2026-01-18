import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def get_expected_output(swiftformer_name):
    if swiftformer_name == 'swiftformer_xs':
        return torch.tensor([-2.1703, 2.1107, -2.0811, 0.88685, 0.2436])
    elif swiftformer_name == 'swiftformer_s':
        return torch.tensor([0.39636, 0.23478, -1.6963, -1.7381, -0.86337])
    elif swiftformer_name == 'swiftformer_l1':
        return torch.tensor([-0.42768, -0.47429, -1.0897, -1.0248, 0.035523])
    elif swiftformer_name == 'swiftformer_l3':
        return torch.tensor([-0.2533, 0.24211, -0.60185, -0.82789, -0.060446])