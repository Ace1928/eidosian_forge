import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
def remove_hook_from_module(module: nn.Module, recurse=False):
    """
    Removes any hook attached to a module via `add_hook_to_module`.

    Args:
        module (`torch.nn.Module`): The module to attach a hook to.
        recurse (`bool`, **optional**): Whether to remove the hooks recursively

    Returns:
        `torch.nn.Module`: The same module, with the hook detached (the module is modified in place, so the result can
        be discarded).
    """
    if hasattr(module, '_hf_hook'):
        module._hf_hook.detach_hook(module)
        delattr(module, '_hf_hook')
    if hasattr(module, '_old_forward'):
        if 'GraphModuleImpl' in str(type(module)):
            module.__class__.forward = module._old_forward
        else:
            module.forward = module._old_forward
        delattr(module, '_old_forward')
    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)
    return module