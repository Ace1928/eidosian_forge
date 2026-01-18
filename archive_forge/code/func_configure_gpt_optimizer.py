import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
@Deprecated(error=False)
def configure_gpt_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, betas: Tuple[float, float]=(0.9, 0.95), **kwargs) -> torch.optim.Optimizer:
    """
    This long function is unfortunately doing something very simple and is
    being very defensive: We are separating out all parameters of the model
    into two buckets: those that will experience weight decay for regularization
    and those that won't (biases, and layernorm/embedding weights). We are then
    returning the PyTorch optimizer object.
    """
    decay = set()
    no_decay = set()
    whitelist_w_modules = (torch.nn.Linear,)
    blacklist_w_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_w_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_w_modules):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f'parameters {str(inter_params)} made it into both decay/no_decay sets!'
    assert len(param_dict.keys() - union_params) == 0, f'parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!'
    optim_groups = [{'params': [param_dict[pn] for pn in sorted(decay)], 'weight_decay': weight_decay}, {'params': [param_dict[pn] for pn in sorted(no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **kwargs)
    return optimizer