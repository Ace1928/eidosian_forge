import argparse
import json
import math
from typing import Tuple
import torch
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils import logging
from transformers.utils.import_utils import is_mamba_ssm_available
def convert_mamba_ssm_checkpoint_to_huggingface_model(original_state_dict: dict, original_ssm_config_dict: dict) -> Tuple[MambaForCausalLM, AutoTokenizer]:
    if not is_mamba_ssm_available():
        raise ImportError('Calling convert_mamba_ssm_checkpoint_to_huggingface_model requires the mamba_ssm library to be installed. Please install it with `pip install mamba_ssm`.')
    original_ssm_config = MambaConfigSSM(**original_ssm_config_dict)
    hf_config = convert_ssm_config_to_hf_config(original_ssm_config)
    converted_state_dict = original_state_dict
    hf_model = MambaForCausalLM(hf_config)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    hf_model.load_state_dict(converted_state_dict)
    return (hf_model, tokenizer)