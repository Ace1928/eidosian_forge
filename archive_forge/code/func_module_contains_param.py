import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized
def module_contains_param(module, parametrization):
    if is_parametrized(module):
        return any((any((isinstance(param, parametrization) for param in param_list)) for key, param_list in module.parametrizations.items()))
    return False