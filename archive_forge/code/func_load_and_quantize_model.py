import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from accelerate.utils.imports import (
from ..big_modeling import dispatch_model, init_empty_weights
from .dataclasses import BnbQuantizationConfig
from .modeling import (
def load_and_quantize_model(model: torch.nn.Module, bnb_quantization_config: BnbQuantizationConfig, weights_location: Union[str, os.PathLike]=None, device_map: Optional[Dict[str, Union[int, str, torch.device]]]=None, no_split_module_classes: Optional[List[str]]=None, max_memory: Optional[Dict[Union[int, str], Union[int, str]]]=None, offload_folder: Optional[Union[str, os.PathLike]]=None, offload_state_dict: bool=False):
    """
    This function will quantize the input model with the associated config passed in `bnb_quantization_config`. If the
    model is in the meta device, we will load and dispatch the weights according to the `device_map` passed. If the
    model is already loaded, we will quantize the model and put the model on the GPU,

    Args:
        model (`torch.nn.Module`):
            Input model. The model can be already loaded or on the meta device
        bnb_quantization_config (`BnbQuantizationConfig`):
            The bitsandbytes quantization parameters
        weights_location (`str` or `os.PathLike`):
            The folder weights_location to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
            - a path to a folder containing a unique pytorch_model.bin file.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit.

    Returns:
        `torch.nn.Module`: The quantized model
    """
    load_in_4bit = bnb_quantization_config.load_in_4bit
    load_in_8bit = bnb_quantization_config.load_in_8bit
    if load_in_8bit and (not is_8bit_bnb_available()):
        raise ImportError('You have a version of `bitsandbytes` that is not compatible with 8bit quantization, make sure you have the latest version of `bitsandbytes` installed.')
    if load_in_4bit and (not is_4bit_bnb_available()):
        raise ValueError('You have a version of `bitsandbytes` that is not compatible with 4bit quantization,make sure you have the latest version of `bitsandbytes` installed.')
    modules_on_cpu = []
    if isinstance(device_map, dict) and len(device_map.keys()) > 1:
        modules_on_cpu = [key for key, value in device_map.items() if value in ['disk', 'cpu']]
    if bnb_quantization_config.skip_modules is None:
        bnb_quantization_config.skip_modules = get_keys_to_not_convert(model)
    if load_in_4bit:
        bnb_quantization_config.skip_modules.extend(modules_on_cpu)
    modules_to_not_convert = bnb_quantization_config.skip_modules
    if bnb_quantization_config.keep_in_fp32_modules is None:
        bnb_quantization_config.keep_in_fp32_modules = []
    keep_in_fp32_modules = bnb_quantization_config.keep_in_fp32_modules
    modules_to_not_convert.extend(keep_in_fp32_modules)
    model.is_loaded_in_4bit = load_in_4bit
    model.is_loaded_in_8bit = load_in_8bit
    model_device = get_parameter_device(model)
    if model_device.type != 'meta':
        logger.warning('It is not recommended to quantize a loaded model. The model should be instantiated under the `init_empty_weights` context manager.')
        model = replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=modules_to_not_convert)
        dtype = bnb_quantization_config.torch_dtype
        for name, param in model.state_dict().items():
            if any((module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules)):
                param.to(torch.float32)
                if param.dtype != torch.float32:
                    name = name.replace('.weight', '').replace('.bias', '')
                    param = getattr(model, name, None)
                    if param is not None:
                        param.to(torch.float32)
            elif torch.is_floating_point(param):
                param.to(dtype)
        if model_device.type == 'cuda':
            model.cuda(torch.cuda.current_device())
            torch.cuda.empty_cache()
        elif torch.cuda.is_available():
            model.to(torch.cuda.current_device())
        else:
            raise RuntimeError('No GPU found. A GPU is needed for quantization.')
        logger.info(f'The model device type is {model_device.type}. However, cuda is needed for quantization.We move the model to cuda.')
        return model
    elif weights_location is None:
        raise RuntimeError(f'`weights_location` needs to be the folder path containing the weights of the model, but we found {weights_location} ')
    else:
        with init_empty_weights():
            model = replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=modules_to_not_convert)
        device_map = get_quantized_model_device_map(model, bnb_quantization_config, device_map, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
        if offload_state_dict is None and device_map is not None and ('disk' in device_map.values()):
            offload_state_dict = True
        offload = any((x in list(device_map.values()) for x in ['cpu', 'disk']))
        load_checkpoint_in_model(model, weights_location, device_map, dtype=bnb_quantization_config.torch_dtype, offload_folder=offload_folder, offload_state_dict=offload_state_dict, keep_in_fp32_modules=bnb_quantization_config.keep_in_fp32_modules, offload_8bit_bnb=load_in_8bit and offload)
        return dispatch_model(model, device_map=device_map, offload_dir=offload_folder)