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
def convert_opus_name_to_hf_name(x):
    """For OPUS-MT-Train/ DEPRECATED"""
    for substr, grp_name in GROUPS:
        x = x.replace(substr, grp_name)
    return x.replace('+', '_')