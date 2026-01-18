import argparse
import io
import json
import os
import tempfile
import urllib
import warnings
from typing import Any, Optional, Tuple
import torch
from huggingface_hub.utils import insecure_hashlib
from torch import nn
from tqdm import tqdm
from transformers import (
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
from transformers.utils.import_utils import _is_package_available
def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)
        print(f'{key} -> {new_key}')
        s_dict[new_key] = s_dict.pop(key)
    return s_dict