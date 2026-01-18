import warnings
import torch
from transformers.pytorch_utils import Conv1D
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
from .gptq import SVDQuantLinear
from .layer import AdaLoraLayer, RankAllocator, SVDLinear
def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name):
    lora_config = self.peft_config[adapter_name]
    for name, rank_idx in rank_pattern.items():
        if isinstance(rank_idx, list):
            rank = sum(rank_idx)
        elif isinstance(rank_idx, torch.Tensor):
            rank_idx = rank_idx.view(-1)
            rank = rank_idx.sum().item()
        else:
            raise ValueError('Unexpected type of rank_idx')
        key = '.'.join(name.split('.')[0:-2]) if adapter_name in name else '.'.join(name.split('.')[0:-1])
        _, target, _ = _get_submodules(self.model, key)
        lora_E_weights = target.lora_E[adapter_name][rank_idx]
        lora_A_weights = target.lora_A[adapter_name][rank_idx]
        lora_B_weights = target.lora_B[adapter_name][:, rank_idx]
        ranknum = target.ranknum[adapter_name]
        target.update_layer(adapter_name, rank, lora_config.lora_alpha, lora_config.lora_dropout, lora_config.init_lora_weights)
        with torch.no_grad():
            if rank > 0:
                target.lora_E[adapter_name].copy_(lora_E_weights)
                target.lora_A[adapter_name].copy_(lora_A_weights)
                target.lora_B[adapter_name].copy_(lora_B_weights)
                target.ranknum[adapter_name].copy_(ranknum)