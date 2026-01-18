import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
def attach_execution_device_hook(module: torch.nn.Module, execution_device: Union[int, str, torch.device], skip_keys: Optional[Union[str, List[str]]]=None, preload_module_classes: Optional[List[str]]=None, tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]]=None):
    """
    Recursively attaches `AlignDevicesHook` to all submodules of a given model to make sure they have the right
    execution device

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`int`, `str` or `torch.device`):
            The device on which inputs and model weights should be placed before the forward pass.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    if not hasattr(module, '_hf_hook') and len(module.state_dict()) > 0:
        add_hook_to_module(module, AlignDevicesHook(execution_device, skip_keys=skip_keys, tied_params_map=tied_params_map))
    if preload_module_classes is not None and module.__class__.__name__ in preload_module_classes:
        return
    for child in module.children():
        attach_execution_device_hook(child, execution_device, tied_params_map=tied_params_map)