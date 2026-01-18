import torch
import copy
from typing import Dict, Any
def set_module_bias(module, bias) -> None:
    if type(module) in _supported_types:
        module.bias = torch.nn.Parameter(bias)
    else:
        module[0].bias = torch.nn.Parameter(bias)