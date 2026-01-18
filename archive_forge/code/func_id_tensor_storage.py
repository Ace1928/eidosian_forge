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
def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.

    This method is the exact same copy of
    https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L282C1-L300C58 but we added
    it here manually to avoid import issue with old versions of transformers.
    """
    if tensor.device.type == 'xla' and is_torch_tpu_available():
        import torch_xla
        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)
    return (tensor.device, unique_id, storage_size(tensor))