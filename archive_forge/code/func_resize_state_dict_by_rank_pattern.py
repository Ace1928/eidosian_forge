import warnings
import torch
from transformers.pytorch_utils import Conv1D
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
from .gptq import SVDQuantLinear
from .layer import AdaLoraLayer, RankAllocator, SVDLinear
def resize_state_dict_by_rank_pattern(self, rank_pattern, state_dict, adapter_name):
    for name, rank_idx in rank_pattern.items():
        rank = sum(rank_idx)
        prefix = '.'.join(name.split('.')[0:-2]) if adapter_name in name else '.'.join(name.split('.')[0:-1])
        for layer in ['lora_E', 'lora_A', 'lora_B']:
            key = f'base_model.model.{prefix}.{layer}.{adapter_name}'
            if layer != 'lora_B':
                state_dict[key] = state_dict[key][rank_idx] if rank != state_dict[key].shape[0] else state_dict[key]
            else:
                state_dict[key] = state_dict[key][:, rank_idx] if rank != state_dict[key].shape[1] else state_dict[key]
    return state_dict