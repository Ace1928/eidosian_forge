import torch
import copy
from typing import Dict, Any
def set_module_weight(module, weight) -> None:
    if type(module) in _supported_types:
        module.weight = torch.nn.Parameter(weight)
    else:
        module[0].weight = torch.nn.Parameter(weight)