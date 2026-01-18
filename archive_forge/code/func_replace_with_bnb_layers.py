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
def replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=None, current_key_name=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules or by `bnb.nn.Linear4bit`
    modules from the `bitsandbytes`library. The function will be run recursively and replace `torch.nn.Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[str]`):
            Names of the modules to not quantize convert. In practice we keep the `lm_head` in full precision for
            numerical stability reasons.
        current_key_name (`List[str]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    model, has_been_replaced = _replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert, current_key_name)
    if not has_been_replaced:
        logger.warning('You are loading your model in 8bit or 4bit but no linear modules were found in your model. this can happen for some architectures such as gpt2 that uses Conv1D instead of Linear layers. Please double check your model architecture, or submit an issue on github if you think this is a bug.')
    return model