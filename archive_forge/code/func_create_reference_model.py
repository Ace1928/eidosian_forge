import json
import logging
import os
from copy import deepcopy
from typing import Optional
import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel
from ..import_utils import is_npu_available, is_peft_available, is_transformers_greater_than, is_xpu_available
def create_reference_model(model: PreTrainedModelWrapper, num_shared_layers: Optional[int]=None, pattern: Optional[str]=None) -> PreTrainedModelWrapper:
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model (`PreTrainedModelWrapper`): The model to be copied.
        num_shared_layers (`int`, *optional*): The number of initial layers that are shared between both models and kept frozen.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns
        `PreTrainedModelWrapper`
    """
    if is_deepspeed_zero3_enabled():
        raise ValueError('DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoCausalLM.from_pretrained()`.')
    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any((pattern_candidate in name for name in parameter_names)):
                pattern = pattern_candidate
                break
    if pattern is None:
        raise ValueError('Layer pattern could not be matched.')
    shared_param_list = []
    unshared_param_list = []
    shared_parameter = True
    for name, _param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False
        _ref_param = ref_model.get_parameter(param_name)
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning('Pattern passed or found, but no layers matched in the model. Check for a typo.')
    return ref_model.eval()