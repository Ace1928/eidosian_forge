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
def set_module_tensor_to_device(module: nn.Module, tensor_name: str, device: Union[int, str, torch.device], value: Optional[torch.Tensor]=None, dtype: Optional[Union[str, torch.dtype]]=None, fp16_statistics: Optional[torch.HalfTensor]=None, tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]]=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
        fp16_statistics (`torch.HalfTensor`, *optional*):
            The list of fp16 statistics to set on the module, used for 8 bit model serialization.
        tied_params_map (Dict[int, Dict[torch.device, torch.Tensor]], *optional*, defaults to `None`):
            A map of current data pointers to dictionaries of devices to already dispatched tied weights. For a given
            execution device, this parameter is useful to reuse the first available pointer of a shared weight on the
            device for all others, instead of duplicating memory.
    """
    if '.' in tensor_name:
        splits = tensor_name.split('.')
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f'{module} has no attribute {split}.')
            module = new_module
        tensor_name = splits[-1]
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f'{module} does not have a parameter or a buffer named {tensor_name}.')
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)
    if value is not None and tied_params_map is not None and (value.data_ptr() in tied_params_map) and (device in tied_params_map[value.data_ptr()]):
        module._parameters[tensor_name] = tied_params_map[value.data_ptr()][device]
        return
    elif tied_params_map is not None and old_value.data_ptr() in tied_params_map and (device in tied_params_map[old_value.data_ptr()]):
        module._parameters[tensor_name] = tied_params_map[old_value.data_ptr()][device]
        return
    if old_value.device == torch.device('meta') and device not in ['meta', torch.device('meta')] and (value is None):
        raise ValueError(f'{tensor_name} is on the meta device, we need a `value` to put in on {device}.')
    if value is not None:
        if old_value.shape != value.shape:
            raise ValueError(f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" (which has shape {old_value.shape}), this look incorrect.')
        if dtype is None:
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(('torch.uint', 'torch.int', 'torch.bool')):
            value = value.to(dtype)
    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)
    device_quantization = None
    with torch.no_grad():
        if param is not None and param.device.type != 'cuda' and (torch.device(device).type == 'cuda') and (param_cls.__name__ in ['Int8Params', 'FP4Params', 'Params4bit']):
            device_quantization = device
            device = 'cpu'
        if is_npu_available() and isinstance(device, int):
            device = f'npu:{device}'
        if is_xpu_available() and isinstance(device, int):
            device = f'xpu:{device}'
        if value is None:
            new_value = old_value.to(device)
            if dtype is not None and device in ['meta', torch.device('meta')]:
                if not str(old_value.dtype).startswith(('torch.uint', 'torch.int', 'torch.bool')):
                    new_value = new_value.to(dtype)
                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ in ['Int8Params', 'FP4Params']:
                if param_cls.__name__ == 'Int8Params' and new_value.dtype == torch.float32:
                    new_value = new_value.to(torch.float16)
                if device == 'cpu' and param_cls.__name__ == 'Int8Params':
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to('cpu')
                    new_value.CB = new_value.CB.to('cpu')
                    new_value.SCB = new_value.SCB.to('cpu')
                else:
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
            elif param_cls.__name__ in ['QTensor', 'QBitsTensor']:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(device)
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
            module._parameters[tensor_name] = new_value
            if fp16_statistics is not None:
                module._parameters[tensor_name].SCB = fp16_statistics.to(device)
                del fp16_statistics
            if module.__class__.__name__ == 'Linear8bitLt' and getattr(module.weight, 'SCB', None) is None and (str(module.weight.device) != 'meta'):
                device_index = torch.device(device).index if torch.device(device).type == 'cuda' else None
                if not getattr(module.weight, 'SCB', None) and device_index is not None:
                    if module.bias is not None and module.bias.device.type != 'meta':
                        module = module.cuda(device_index)
                    elif module.bias is None:
                        module = module.cuda(device_index)
            elif module.__class__.__name__ == 'Linear4bit' and getattr(module.weight, 'quant_state', None) is None:
                device_index = torch.device(device).index if torch.device(device).type == 'cuda' else None
                if not getattr(module.weight, 'quant_state', None) and device_index is not None:
                    module.weight = module.weight.cuda(device_index)
    if is_npu_available():
        torch.npu.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()
    if tied_params_map is not None and old_value.data_ptr() in tied_params_map and (device not in tied_params_map[old_value.data_ptr()]):
        tied_params_map[old_value.data_ptr()][device] = new_value
    elif value is not None and tied_params_map is not None and (value.data_ptr() in tied_params_map) and (device not in tied_params_map[value.data_ptr()]):
        tied_params_map[value.data_ptr()][device] = new_value