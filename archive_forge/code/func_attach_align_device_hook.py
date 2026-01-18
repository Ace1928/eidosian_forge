import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
def attach_align_device_hook(module: torch.nn.Module, execution_device: Optional[torch.device]=None, offload: bool=False, weights_map: Optional[Mapping]=None, offload_buffers: bool=False, module_name: str='', skip_keys: Optional[Union[str, List[str]]]=None, preload_module_classes: Optional[List[str]]=None, tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]]=None):
    """
    Recursively attaches `AlignDevicesHook` to all submodules of a given model that have direct parameters and/or
    buffers.

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`torch.device`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        module_name (`str`, *optional*, defaults to `""`):
            The name of the module.
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
    directs = named_module_tensors(module)
    full_offload = offload and preload_module_classes is not None and (module.__class__.__name__ in preload_module_classes)
    if len(list(directs)) > 0 or full_offload:
        if weights_map is not None:
            prefix = f'{module_name}.' if len(module_name) > 0 else ''
            prefixed_weights_map = PrefixedDataset(weights_map, prefix)
        else:
            prefixed_weights_map = None
        hook = AlignDevicesHook(execution_device=execution_device, offload=offload, weights_map=prefixed_weights_map, offload_buffers=offload_buffers, place_submodules=full_offload, skip_keys=skip_keys, tied_params_map=tied_params_map)
        add_hook_to_module(module, hook, append=True)
    if full_offload:
        return
    for child_name, child in module.named_children():
        child_name = f'{module_name}.{child_name}' if len(module_name) > 0 else child_name
        attach_align_device_hook(child, execution_device=execution_device, offload=offload, weights_map=weights_map, offload_buffers=offload_buffers, module_name=child_name, preload_module_classes=preload_module_classes, skip_keys=skip_keys, tied_params_map=tied_params_map)