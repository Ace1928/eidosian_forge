import logging
from typing import Tuple
from torch import nn
def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule('.'.join(module_name.split('.')[:-1]))
    target_name = module_name.split('.')[-1]
    setattr(parent, target_name, new_module)
    return new_module