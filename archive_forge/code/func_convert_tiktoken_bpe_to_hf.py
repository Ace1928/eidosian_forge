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
def convert_tiktoken_bpe_to_hf(tiktoken_url: str):
    bpe_ranks = load_tiktoken_bpe(tiktoken_url)
    byte_encoder = bytes_to_unicode()

    def token_bytes_to_string(b):
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])
    merges = []
    vocab = {}
    for token, rank in bpe_ranks.items():
        vocab[token_bytes_to_string(token)] = rank
        if len(token) == 1:
            continue
        merged = tuple(_bpe(bpe_ranks, token, max_rank=rank))
        if len(merged) == 2:
            merges.append(' '.join(map(token_bytes_to_string, merged)))
    return (vocab, merges)