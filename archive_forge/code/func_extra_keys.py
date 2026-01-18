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
@property
def extra_keys(self):
    extra = []
    for k in self.state_keys:
        if k.startswith('encoder_l') or k.startswith('decoder_l') or k in [CONFIG_KEY, 'Wemb', 'encoder_Wemb', 'decoder_Wemb', 'Wpos', 'decoder_ff_logit_out_b']:
            continue
        else:
            extra.append(k)
    return extra