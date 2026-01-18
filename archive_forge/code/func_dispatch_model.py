import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from .hooks import (
from .utils import (
from .utils.other import recursive_getattr
def dispatch_model(model: nn.Module, device_map: Dict[str, Union[str, int, torch.device]], main_device: Optional[torch.device]=None, state_dict: Optional[Dict[str, torch.Tensor]]=None, offload_dir: Optional[Union[str, os.PathLike]]=None, offload_index: Optional[Dict[str, str]]=None, offload_buffers: bool=False, skip_keys: Optional[Union[str, List[str]]]=None, preload_module_classes: Optional[List[str]]=None, force_hooks: bool=False):
    """
    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype`/ `shape` or safetensors filename). Will default
            to the index saved in `save_folder`.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
    """
    check_device_map(model, device_map)
    is_bnb_quantized = (getattr(model, 'is_quantized', False) or getattr(model, 'is_loaded_in_8bit', False)) and getattr(model, 'quantization_method', 'bitsandbytes') == 'bitsandbytes'
    if len(set(device_map.values())) > 1 or is_bnb_quantized or force_hooks:
        if main_device is None:
            if set(device_map.values()) == {'cpu'} or set(device_map.values()) == {'cpu', 'disk'}:
                main_device = 'cpu'
            else:
                main_device = [d for d in device_map.values() if d not in ['cpu', 'disk']][0]
        if main_device != 'cpu':
            cpu_modules = [name for name, device in device_map.items() if device == 'cpu']
            if state_dict is None and len(cpu_modules) > 0:
                state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)
        disk_modules = [name for name, device in device_map.items() if device == 'disk']
        if offload_dir is None and offload_index is None and (len(disk_modules) > 0):
            raise ValueError(f'We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules need to be offloaded: {', '.join(disk_modules)}.')
        if len(disk_modules) > 0 and offload_index is None and (not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, 'index.json'))):
            disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
            offload_state_dict(offload_dir, disk_state_dict)
        execution_device = {name: main_device if device in ['cpu', 'disk'] else device for name, device in device_map.items()}
        execution_device[''] = main_device
        offloaded_devices = ['disk'] if main_device == 'cpu' or main_device == 'mps' else ['cpu', 'disk']
        offload = {name: device in offloaded_devices for name, device in device_map.items()}
        save_folder = offload_dir if len(disk_modules) > 0 else None
        if state_dict is not None or save_folder is not None or offload_index is not None:
            device = main_device if offload_index is not None else None
            weights_map = OffloadedWeightsLoader(state_dict=state_dict, save_folder=save_folder, index=offload_index, device=device)
        else:
            weights_map = None
        tied_params = find_tied_parameters(model)
        tied_params_map = {}
        for group in tied_params:
            for param_name in group:
                data_ptr = recursive_getattr(model, param_name).data_ptr()
                tied_params_map[data_ptr] = {}
        attach_align_device_hook_on_blocks(model, execution_device=execution_device, offload=offload, offload_buffers=offload_buffers, weights_map=weights_map, skip_keys=skip_keys, preload_module_classes=preload_module_classes, tied_params_map=tied_params_map)
        offloaded_devices_str = ' and '.join([device for device in set(device_map.values()) if device in ('cpu', 'disk')])
        if len(offloaded_devices_str) > 0:
            logging.warning(f'Some parameters are on the meta device device because they were offloaded to the {offloaded_devices_str}.')
        retie_parameters(model, tied_params)

        def add_warning(fn, model):

            @wraps(fn)
            def wrapper(*args, **kwargs):
                warning_msg = "You shouldn't move a model that is dispatched using accelerate hooks."
                if str(fn.__name__) == 'to':
                    to_device = torch._C._nn._parse_to(*args, **kwargs)[0]
                    if to_device is not None:
                        logger.warning(warning_msg)
                else:
                    logger.warning(warning_msg)
                for param in model.parameters():
                    if param.device == torch.device('meta'):
                        raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
                return fn(*args, **kwargs)
            return wrapper
        model.to = add_warning(model.to, model)
        if is_npu_available():
            model.npu = add_warning(model.npu, model)
        elif is_xpu_available():
            model.xpu = add_warning(model.xpu, model)
        else:
            model.cuda = add_warning(model.cuda, model)
    else:
        device = list(device_map.values())[0]
        if is_npu_available() and isinstance(device, int):
            device = f'npu:{device}'
        elif is_xpu_available() and isinstance(device, int):
            device = f'xpu:{device}'
        if device != 'disk':
            model.to(device)
        else:
            raise ValueError('You are trying to offload the whole model to the disk. Please use the `disk_offload` function instead.')
    model.hf_device_map = dict(device_map)
    return model