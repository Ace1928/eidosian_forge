import argparse
import glob
import json
from pathlib import Path
import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open
from transformers import (
def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=['*.safetensors'])
    original_state_dict = {}
    for path in glob.glob(f'{directory_path}/*'):
        if path.endswith('.safetensors'):
            with safe_open(path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)
    return original_state_dict