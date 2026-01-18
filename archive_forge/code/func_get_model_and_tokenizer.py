from itertools import product
import math
import pytest
import torch
import transformers
from transformers import (
from tests.helpers import TRUE_FALSE, describe_dtype, id_formatter
def get_model_and_tokenizer(config):
    model_name_or_path, quant_type = config
    bnb_config = get_4bit_config()
    if quant_type == '16bit':
        bnb_config.load_in_4bit = False
    else:
        bnb_config.bnb_4bit_quant_type = quant_type
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config, max_memory={0: '48GB'}, device_map='auto', torch_dtype=torch.bfloat16).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    return (model, tokenizer)