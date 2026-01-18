import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple
import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size
from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
def check_module(self):
    """Perform some sanity checks on the module to ensure that it works"""
    forbidden_classes = (torch.nn.ModuleDict, torch.nn.ModuleList, torch.nn.ParameterDict, torch.nn.ParameterList)
    if isinstance(self.original_module, forbidden_classes):
        cls_name = self.original_module.__class__.__name__
        raise TypeError(f'modules_to_save cannot be applied to modules of type {cls_name}')