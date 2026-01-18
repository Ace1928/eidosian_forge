import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
def check_marian_cfg_assumptions(marian_cfg):
    assumed_settings = {'layer-normalization': False, 'right-left': False, 'transformer-ffn-depth': 2, 'transformer-aan-depth': 2, 'transformer-no-projection': False, 'transformer-postprocess-emb': 'd', 'transformer-postprocess': 'dan', 'transformer-preprocess': '', 'type': 'transformer', 'ulr-dim-emb': 0, 'dec-cell-base-depth': 2, 'dec-cell-high-depth': 1, 'transformer-aan-nogate': False}
    for k, v in assumed_settings.items():
        actual = marian_cfg[k]
        if actual != v:
            raise ValueError(f'Unexpected config value for {k} expected {v} got {actual}')