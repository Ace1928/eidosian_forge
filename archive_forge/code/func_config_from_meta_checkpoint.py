import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Config, LlamaConfig
from einops import rearrange
def config_from_meta_checkpoint(checkpoint_path: Union[str, os.PathLike], model_name: str) -> LlamaConfig:
    """Load a LlamaConfig from a checkpoint path."""
    with open(Path(checkpoint_path) / model_name / 'params.json') as f:
        params = json.load(f)
    config = LlamaConfig(hidden_size=params['dim'], intermediate_size=None, num_attention_heads=params['n_heads'], num_hidden_layers=params['n_layers'], rms_norm_eps=params['norm_eps'], num_key_value_heads=params.get('n_kv_heads', None))
    multiple_of = params.get('multiple_of', 1)
    ffn_dim_multiplier = params.get('ffn_dim_multiplier', None)
    intermediate_size = 4 * config.hidden_size
    intermediate_size = int(2 * intermediate_size / 3)
    if ffn_dim_multiplier is not None:
        intermediate_size = int(ffn_dim_multiplier * intermediate_size)
    intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
    config.intermediate_size = intermediate_size
    if 'rope_theta' in params:
        config.rotary_emb_base = params['rope_theta']
    config.vocab_size = 32000
    tokenizer = Path(checkpoint_path) / model_name / 'tokenizer.model'
    if tokenizer.is_file():
        config.vocab_size = SentencePieceProcessor(str(tokenizer)).vocab_size()
    return config