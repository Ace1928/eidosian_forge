import contextlib
import gc
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import packaging
import torch
import torch.nn as nn
from ..state import AcceleratorState
from .constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from .dataclasses import AutocastKwargs, CustomDtype, DistributedType
from .imports import is_mps_available, is_npu_available, is_peft_available, is_torch_xla_available, is_xpu_available
from .offload import load_offloaded_weight, offload_weight, save_offload_index
from .tqdm import is_tqdm_available, tqdm
from .versions import compare_versions
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
def get_state_dict_offloaded_model(model: nn.Module):
    """
    Returns the state dictionary for an offloaded model via iterative onloading

    Args:
        model (`torch.nn.Module`):
            The offloaded model we want to save
    """
    from ..hooks import AlignDevicesHook
    state_dict = {}
    placeholders = set()
    for name, module in model.named_modules():
        if name == '':
            continue
        if hasattr(module, '_hf_hook') and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            original_device = module._hf_hook.execution_device
            module._hf_hook.execution_device = 'cpu'
            try:
                module._hf_hook.pre_forward(module)
            except MemoryError:
                raise MemoryError('Offloaded module must fit in CPU memory to call save_model!') from None
            module_state_dict = module.state_dict()
            module._hf_hook.post_forward(module, torch.tensor([]))
            module._hf_hook.execution_device = original_device
        else:
            module_state_dict = module.state_dict()
        for key in module_state_dict:
            if module_state_dict[key].device == torch.device('meta'):
                placeholders.add(name + f'.{key}')
                continue
            params = module_state_dict[key]
            state_dict[name + f'.{key}'] = params
    for key in placeholders.copy():
        if key in state_dict:
            placeholders.remove(key)
    if placeholders:
        logger.warning(f'The following tensors were not saved because they were still on meta device: {placeholders}')
    return state_dict