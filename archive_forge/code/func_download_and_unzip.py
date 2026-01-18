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
def download_and_unzip(url, dest_dir):
    try:
        import wget
    except ImportError:
        raise ImportError('you must pip install wget')
    filename = wget.download(url)
    unzip(filename, dest_dir)
    os.remove(filename)