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
def convert_hf_name_to_opus_name(hf_model_name):
    """
    Relies on the assumption that there are no language codes like pt_br in models that are not in GROUP_TO_OPUS_NAME.
    """
    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    if hf_model_name in GROUP_TO_OPUS_NAME:
        opus_w_prefix = GROUP_TO_OPUS_NAME[hf_model_name]
    else:
        opus_w_prefix = hf_model_name.replace('_', '+')
    return remove_prefix(opus_w_prefix, 'opus-mt-')