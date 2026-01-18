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
def infer_auto_device_map(model: nn.Module, max_memory: Optional[Dict[Union[int, str], Union[int, str]]]=None, no_split_module_classes: Optional[List[str]]=None, dtype: Optional[Union[str, torch.dtype]]=None, special_dtypes: Optional[Dict[str, Union[str, torch.dtype]]]=None, verbose: bool=False, clean_result: bool=True, offload_buffers: bool=False):
    """
    Compute a device map for a given model giving priority to GPUs, then offload on CPU and finally offload to disk,
    such that:
    - we don't exceed the memory available of any of the GPU.
    - if offload to the CPU is needed, there is always room left on GPU 0 to put back the layer offloaded on CPU that
      has the largest size.
    - if offload to the CPU is needed,we don't exceed the RAM available on the CPU.
    - if offload to the disk is needed, there is always room left on the CPU to put back the layer offloaded on disk
      that has the largest size.

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
        verbose (`bool`, *optional*, defaults to `False`):
            Whether or not to provide debugging statements as the function builds the device_map.
        clean_result (`bool`, *optional*, defaults to `True`):
            Clean the resulting device_map by grouping all submodules that go on the same device together.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
    """
    max_memory = get_max_memory(max_memory)
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]
    devices = list(max_memory.keys())
    if 'disk' not in devices:
        devices.append('disk')
    gpus = [device for device in devices if device not in ['cpu', 'disk']]
    if 'mps' in gpus:
        main_devices = ['mps']
    elif len(gpus) > 0:
        main_devices = [gpus[0], 'cpu']
    else:
        main_devices = ['cpu']
    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    tied_parameters = find_tied_parameters(model)
    if check_tied_parameters_in_config(model) and len(tied_parameters) == 0:
        logger.warn('The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.')
    device_map = OrderedDict()
    current_device = 0
    current_memory_used = 0
    device_memory_used = {}
    device_buffer_sizes = {}
    modules_to_treat = list(model.named_parameters(recurse=False)) + list(model.named_children()) + list(model.named_buffers(recurse=False))
    max_layer_size, max_layer_names = get_max_layer_size(modules_to_treat, module_sizes, no_split_module_classes)
    while len(modules_to_treat) > 0:
        name, module = modules_to_treat.pop(0)
        if verbose:
            print(f'\nTreating module {name}.')
        max_layer_names = [n for n in max_layer_names if n != name and (not n.startswith(name + '.'))]
        if len(max_layer_names) == 0:
            max_layer_size, max_layer_names = get_max_layer_size([(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)], module_sizes, no_split_module_classes)
        module_size = module_sizes[name]
        tied_param_goups = [tied_group for tied_group in tied_parameters if any((name + '.' in k + '.' for k in tied_group)) and (not all((name + '.' in k + '.' for k in tied_group)))]
        if verbose and len(tied_param_goups) > 0:
            print(f'  Found the relevant tied param groups {tied_param_goups}')
        tied_params = sum([[p for p in tied_group if name + '.' not in p + '.'] for tied_group in tied_param_goups], [])
        if verbose and len(tied_params) > 0:
            print(f'  So those parameters need to be taken into account {tied_params}')
        device = devices[current_device]
        current_max_size = max_memory[device] if device != 'disk' else None
        current_memory_reserved = 0
        if devices[current_device] in main_devices:
            current_max_size = current_max_size - max_layer_size
            current_memory_reserved = max_layer_size
        if current_max_size is not None and current_memory_used + module_size > current_max_size:
            modules_children = [] if isinstance(module, nn.Parameter) or isinstance(module, torch.Tensor) else list(module.named_children())
            if verbose:
                print(f'Not enough space on {devices[current_device]} to put {name} (space available {current_max_size - current_memory_used}, module size {module_size}).')
            if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
                if verbose:
                    print('This module cannot be split, going to the next device.')
                device_memory_used[device] = current_memory_used + current_memory_reserved
                current_device += 1
                modules_to_treat = [(name, module)] + modules_to_treat
                current_memory_used = 0
            else:
                if verbose:
                    print(f'Splitting {name}.')
                modules_children = list(module.named_parameters(recurse=False)) + modules_children
                modules_to_treat = [(f'{name}.{n}', v) for n, v in modules_children] + modules_to_treat
                max_layer_size, max_layer_names = get_max_layer_size([(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)], module_sizes, no_split_module_classes)
        elif len(tied_params) > 0:
            tied_module_names = []
            tied_modules = []
            for tied_param in tied_params:
                tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n in tied_param][0]
                tied_module_names.append(modules_to_treat[tied_module_index][0])
                tied_modules.append(modules_to_treat[tied_module_index][1])
            if verbose:
                print(f'  It looks like {name} is going to fit on {devices[current_device]} but we have tied parameters to account for.\n  - Names {tied_params}\n  - Module names {tied_module_names}')
            module_size_with_ties = module_size
            for tied_param, tied_module_name in zip(tied_params, tied_module_names):
                module_size_with_ties += module_sizes[tied_module_name] - module_sizes[tied_param]
            if current_max_size is None or current_memory_used + module_size_with_ties <= current_max_size:
                if verbose:
                    print(f'Putting {name} and {tied_module_names} on {devices[current_device]}.')
                current_memory_used += module_size_with_ties
                device_map[name] = devices[current_device]
                for tied_module_name in tied_module_names:
                    if tied_module_name in [m[0] for m in modules_to_treat]:
                        tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]
                        modules_to_treat.pop(tied_module_index)
                    device_map[tied_module_name] = devices[current_device]
                if not offload_buffers and isinstance(module, nn.Module):
                    current_buffer_size = compute_module_total_buffer_size(module, dtype=dtype, special_dtypes=special_dtypes)
                    device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size
            else:
                if verbose:
                    print(f'Not enough space on {devices[current_device]} to put {name} and {tied_module_names} (space available {current_max_size - current_memory_used}, needed size {module_size_with_ties}).')
                split_happened = False
                for tied_module_name, tied_module in zip(tied_module_names, tied_modules):
                    tied_module_children = list(tied_module.named_children())
                    if len(tied_module_children) == 0 or tied_module.__class__.__name__ in no_split_module_classes:
                        continue
                    if verbose:
                        print(f'Splitting {tied_module_name}.')
                    tied_module_children = list(tied_module.named_parameters(recurse=False)) + tied_module_children
                    tied_module_children = [(f'{tied_module_name}.{n}', v) for n, v in tied_module_children]
                    tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]
                    modules_to_treat = [(name, module)] + modules_to_treat[:tied_module_index] + tied_module_children + modules_to_treat[tied_module_index + 1:]
                    max_layer_size, max_layer_names = get_max_layer_size([(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)], module_sizes, no_split_module_classes)
                    split_happened = True
                    break
                if not split_happened:
                    if verbose:
                        print('None of the tied module can be split, going to the next device.')
                    device_memory_used[device] = current_memory_used + current_memory_reserved
                    current_device += 1
                    modules_to_treat = [(name, module)] + modules_to_treat
                    current_memory_used = 0
        else:
            if verbose:
                if current_max_size is None:
                    print(f'Putting {name} (size={module_size}) on {devices[current_device]}.')
                else:
                    print(f'Putting {name} (size={module_size}) on {devices[current_device]} (available={current_max_size - current_memory_used}).')
            current_memory_used += module_size
            device_memory_used[device] = current_memory_used + current_memory_reserved
            device_map[name] = devices[current_device]
            if not offload_buffers and isinstance(module, nn.Module):
                current_buffer_size = compute_module_total_buffer_size(module, dtype=dtype, special_dtypes=special_dtypes)
                device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size
    if clean_result:
        device_map = clean_device_map(device_map)
    non_gpu_buffer_size = device_buffer_sizes.get('cpu', 0) + device_buffer_sizes.get('disk', 0)
    if non_gpu_buffer_size > 0 and (not offload_buffers):
        is_buffer_fit_any_gpu = False
        for gpu_device, gpu_max_memory in max_memory.items():
            if gpu_device == 'cpu' or gpu_device == 'disk':
                continue
            if not is_buffer_fit_any_gpu:
                gpu_memory_used = device_memory_used.get(gpu_device, 0)
                if gpu_max_memory >= non_gpu_buffer_size + gpu_memory_used:
                    is_buffer_fit_any_gpu = True
        if len(gpus) > 0 and (not is_buffer_fit_any_gpu):
            warnings.warn(f"Current model requires {non_gpu_buffer_size} bytes of buffer for offloaded layers, which seems does not fit any GPU's remaining memory. If you are experiencing a OOM later, please consider using offload_buffers=True.")
    return device_map