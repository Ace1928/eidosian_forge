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
def get_balanced_memory(model: nn.Module, max_memory: Optional[Dict[Union[int, str], Union[int, str]]]=None, no_split_module_classes: Optional[List[str]]=None, dtype: Optional[Union[str, torch.dtype]]=None, special_dtypes: Optional[Dict[str, Union[str, torch.device]]]=None, low_zero: bool=False):
    """
    Compute a `max_memory` dictionary for [`infer_auto_device_map`] that will balance the use of each available GPU.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
            Example: `max_memory={0: "1GB"}`.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, Union[str, torch.device]]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        low_zero (`bool`, *optional*):
            Minimizes the number of weights on GPU 0, which is convenient when it's used for other operations (like the
            Transformers generate function).
    """
    user_not_set_max_memory = max_memory is None
    max_memory = get_max_memory(max_memory)
    if is_npu_available():
        num_devices = len([d for d in max_memory if torch.device(d).type == 'npu' and max_memory[d] > 0])
    elif is_xpu_available():
        num_devices = len([d for d in max_memory if (d != 'cpu' and (torch.device(d).type == 'xpu' or torch.xpu.get_device_properties(d).dev_type == 'gpu')) and max_memory[d] > 0])
    else:
        num_devices = len([d for d in max_memory if torch.device(d).type == 'cuda' and max_memory[d] > 0])
    if num_devices == 0:
        return max_memory
    if num_devices == 1:
        low_zero = False
        if user_not_set_max_memory:
            for key in max_memory.keys():
                if isinstance(key, int):
                    max_memory[key] *= 0.9
                    logger.info(f'We will use 90% of the memory on device {key} for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).')
                    break
    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    per_gpu = module_sizes[''] // (num_devices - 1 if low_zero else num_devices)
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]
    if len(no_split_module_classes) > 0:
        no_split_children = {}
        for name, size in module_sizes.items():
            if name == '':
                continue
            submodule = model
            for submodule_name in name.split('.'):
                submodule = getattr(submodule, submodule_name)
            class_name = submodule.__class__.__name__
            if class_name in no_split_module_classes and class_name not in no_split_children:
                no_split_children[class_name] = size
            if set(no_split_children.keys()) == set(no_split_module_classes):
                break
        buffer = max(no_split_children.values()) if len(no_split_children) > 0 else 0
    else:
        buffer = 0
    leaves = [n for n in module_sizes if len([p for p in module_sizes if n == '' or p.startswith(n + '.')]) == 0]
    module_sizes = {n: v for n, v in module_sizes.items() if n not in leaves}
    leaves = [n for n in module_sizes if len([p for p in module_sizes if n == '' or p.startswith(n + '.')]) == 0]
    mean_leaves = int(sum([module_sizes[n] for n in leaves]) / max(len(leaves), 1))
    buffer = int(1.25 * max(buffer, mean_leaves))
    per_gpu += buffer
    gpus_idx_list = list(sorted((device_id for device_id, device_mem in max_memory.items() if isinstance(device_id, int) and device_mem > 0)))
    for idx in gpus_idx_list[:-1]:
        max_memory[idx] = min(max_memory[0] if low_zero and idx == 0 else per_gpu, max_memory[idx])
    if low_zero:
        min_zero = max(0, module_sizes[''] - sum([max_memory[i] for i in range(1, num_devices)]))
        max_memory[0] = min(min_zero, max_memory[0])
    return max_memory