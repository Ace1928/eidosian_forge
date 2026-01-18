from typing import Any, Dict, Optional, Type
from torch.nn.utils.parametrize import type_before_parametrizations, is_parametrized
from itertools import chain
from torch import nn
def module_to_fqn(model: nn.Module, module: nn.Module, prefix: str='') -> Optional[str]:
    """
    Returns the fqn for a module or None if module not a descendent of model.
    """
    if module is model:
        return ''
    for name, child in model.named_children():
        fqn = module_to_fqn(child, module, '.')
        if isinstance(fqn, str):
            return prefix + name + fqn
    return None