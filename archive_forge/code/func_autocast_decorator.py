import functools
import warnings
from typing import Any, Optional
import torch
from torch.types import _dtype
def autocast_decorator(autocast_instance, func):

    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)
    decorate_autocast.__script_unsupported = '@autocast() decorator is not supported in script mode'
    return decorate_autocast