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
def fetch_test_set(test_set_url):
    import wget
    fname = wget.download(test_set_url, 'opus_test.txt')
    lns = Path(fname).open().readlines()
    src = lmap(str.strip, lns[::4])
    gold = lmap(str.strip, lns[1::4])
    mar_model = lmap(str.strip, lns[2::4])
    if not len(gold) == len(mar_model) == len(src):
        raise ValueError(f'Gold, marian and source lengths {len(gold)}, {len(mar_model)}, {len(src)} mismatched')
    os.remove(fname)
    return (src, mar_model, gold)