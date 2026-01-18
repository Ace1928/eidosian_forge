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
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer