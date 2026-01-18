from typing import Any
import torch
from torch._utils import _get_device_index as _torch_get_device_index
def err_fn(obj, *args, **kwargs):
    if is_init:
        class_name = obj.__class__.__name__
    else:
        class_name = obj.__name__
    raise RuntimeError(f'Tried to instantiate dummy base class {class_name}')