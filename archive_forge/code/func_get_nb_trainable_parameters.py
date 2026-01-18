from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import remove_hook_from_submodules
from torch import nn
from transformers.utils import PushToHubMixin
from peft.tuners.mixed import COMPATIBLE_TUNER_TYPES
from .config import PeftConfig
from .peft_model import PeftModel
from .tuners import (
from .utils import PeftType, _set_adapter, _set_trainable
def get_nb_trainable_parameters(self):
    """
        Returns the number of trainable parameters and number of all parameters in the model.
        """
    trainable_params = 0
    all_param = 0
    for _, param in self.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, 'ds_numel'):
            num_params = param.ds_numel
        if param.__class__.__name__ == 'Params4bit':
            num_params = num_params * 2
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return (trainable_params, all_param)