import argparse
import collections
import json
from pathlib import Path
import requests
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def remove_unused_keys(state_dict):
    """remove unused keys (e.g.: seg_head.aux_head)"""
    keys_to_ignore = []
    for k in state_dict.keys():
        if k.startswith('seg_head.aux_head.'):
            keys_to_ignore.append(k)
    for k in keys_to_ignore:
        state_dict.pop(k, None)