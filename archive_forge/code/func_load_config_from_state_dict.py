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
def load_config_from_state_dict(opus_dict):
    import yaml
    cfg_str = ''.join([chr(x) for x in opus_dict[CONFIG_KEY]])
    yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)
    return cast_marian_config(yaml_cfg)