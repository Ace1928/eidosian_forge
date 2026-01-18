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
def quantize_and_offload_8bit(model, param, param_name, new_dtype, offload_folder, offload_index, fp16_statistics):
    if fp16_statistics is None:
        set_module_tensor_to_device(model, param_name, 0, dtype=new_dtype, value=param)
        tensor_name = param_name
        module = model
        if '.' in tensor_name:
            splits = tensor_name.split('.')
            for split in splits[:-1]:
                new_module = getattr(module, split)
                if new_module is None:
                    raise ValueError(f'{module} has no attribute {split}.')
                module = new_module
            tensor_name = splits[-1]
        module._parameters[tensor_name].requires_grad = False
        offload_weight(module._parameters[tensor_name], param_name, offload_folder, index=offload_index)
        if hasattr(module._parameters[tensor_name], 'SCB'):
            offload_weight(module._parameters[tensor_name].SCB, param_name.replace('weight', 'SCB'), offload_folder, index=offload_index)
    else:
        offload_weight(param, param_name, offload_folder, index=offload_index)
        offload_weight(fp16_statistics, param_name.replace('weight', 'SCB'), offload_folder, index=offload_index)
    set_module_tensor_to_device(model, param_name, 'meta', dtype=new_dtype, value=torch.empty(*param.size()))