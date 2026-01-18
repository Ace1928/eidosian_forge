import argparse
import json
import math
from typing import Tuple
import torch
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils import logging
from transformers.utils.import_utils import is_mamba_ssm_available
def convert_mamba_checkpoint_file_to_huggingface_model_file(mamba_checkpoint_path: str, config_json_file: str, output_dir: str) -> None:
    if not is_mamba_ssm_available():
        raise ImportError('Calling convert_mamba_checkpoint_file_to_huggingface_model_file requires the mamba_ssm library to be installed. Please install it with `pip install mamba_ssm`.')
    if not torch.cuda.is_available():
        raise ValueError('This script is to be run with a CUDA device, as the original mamba_ssm model does not support cpu.')
    logger.info(f'Loading model from {mamba_checkpoint_path} based on config from {config_json_file}')
    original_state_dict = torch.load(mamba_checkpoint_path, map_location='cpu')
    with open(config_json_file, 'r', encoding='utf-8') as json_file:
        original_ssm_config_dict = json.load(json_file)
    hf_model, tokenizer = convert_mamba_ssm_checkpoint_to_huggingface_model(original_state_dict, original_ssm_config_dict)
    validate_converted_model(original_state_dict, original_ssm_config_dict, hf_model, tokenizer)
    logger.info(f'Model converted successfully. Saving model to {output_dir}')
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)