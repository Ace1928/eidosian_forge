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
def cast_mixed_precision_params(model, dtype):
    """
    Cast all non-trainable parameters of the model to the given `dtype`. The `dtype` can be `torch.float16` or
    `torch.bfloat16` as per the mixed-precision training you are performing. The trainable parameters are cast to full
    precision. This is meant to reduce the GPU memory usage when using PEFT methods by using half-precision dtype for
    non-trainable parameters. Having the trainable parameters in full-precision preserves training stability when using
    automatic mixed-precision training.

    Args:
        model (`torch.nn.Module`):
            The model to cast the non-trainable parameters of.
        dtype (`torch.dtype`):
            The dtype to cast the non-trainable parameters to. The `dtype` can be `torch.float16` or
    `torch.bfloat16` as per the mixed-precision training you are performing.
    """
    for p in model.parameters():
        if not p.requires_grad:
            p.data = p.to(dtype)
        else:
            p.data = p.to(torch.float32)