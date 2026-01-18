import argparse
import json
import math
from typing import Tuple
import torch
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils import logging
from transformers.utils.import_utils import is_mamba_ssm_available
def convert_ssm_config_to_hf_config(config_ssm: MambaConfigSSM) -> MambaConfig:
    """Convert a MambaConfig from mamba_ssm to a MambaConfig from transformers."""
    hf_config = MambaConfig()
    hf_config.hidden_size = config_ssm.d_model
    hf_config.intermediate_size = config_ssm.d_model * 2
    hf_config.time_step_rank = math.ceil(config_ssm.d_model / 16)
    hf_config.num_hidden_layers = config_ssm.n_layer
    vocab_size = config_ssm.vocab_size
    pad_vocab_size_multiple = config_ssm.pad_vocab_size_multiple
    if vocab_size % pad_vocab_size_multiple != 0:
        vocab_size += pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple
    hf_config.vocab_size = vocab_size
    return hf_config