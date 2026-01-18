import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
def mask_using_rank_pattern(self, model, rank_pattern):
    is_adapter_name_truncated = False
    if self.adapter_name not in next(iter(rank_pattern.keys())):
        is_adapter_name_truncated = True
    with torch.no_grad():
        for n, p in model.named_parameters():
            if f'lora_E.{self.adapter_name}' in n:
                key = n if not is_adapter_name_truncated else n.replace(f'.{self.adapter_name}', '')
                mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                p.masked_fill_(~mask.bool(), 0.0)