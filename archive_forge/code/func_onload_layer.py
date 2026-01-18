from __future__ import annotations
import logging
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
@contextmanager
def onload_layer(layer):
    """
    A utility for modifying a module containing one or more tuners and a base layer, any of which are offloaded to the
    CPU or disk. Moves a module's sub-modules to the execution device before some action is performed, after that the
    base layer state dictionary is re-assigned (if that layer was offloaded to the disk) and finally the parameters are
    offloaded.

    If the module has no offloaded sub-modules, this function does nothing.

    Args:
        layer ('torch.nn.Module'):
            layer with tuners to be merged
    """
    offloaded_modules = []
    for name, module in layer.named_modules():
        if name in ['', 'base_layer']:
            continue
        if hasattr(module, '_hf_hook') and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            module._hf_hook.pre_forward(module)
            offloaded_modules.append(module)
    base_layer_offload = False
    if hasattr(layer, 'base_layer') and (hasattr(layer.base_layer, '_hf_hook') and isinstance(layer.base_layer._hf_hook, AlignDevicesHook) and layer.base_layer._hf_hook.offload):
        if torch.device('meta') in layer.base_layer._hf_hook.original_devices.values():
            offload_folder = layer.base_layer._hf_hook.weights_map.dataset.save_folder
        layer.base_layer._hf_hook.pre_forward(layer.base_layer)
        base_layer_offload = True
    yield
    for module in offloaded_modules:
        module._hf_hook.post_forward(module, torch.tensor([]))
    if base_layer_offload:
        layer.base_layer._hf_hook.weights_map = {name: param.to('cpu') for name, param in named_module_tensors(layer.base_layer)}
        if torch.device('meta') in layer.base_layer._hf_hook.original_devices.values():
            offload_state_dict(offload_folder, layer.base_layer._hf_hook.weights_map)
        layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))