from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def check_same_device(*args, allow_cpu_scalar_tensors):
    """
    Checks that all Tensors in args have the same device.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices, unless one is a CPU scalar tensor and allow_cpu_scalar_tensors is True
    """
    if len(args) <= 1:
        return
    device = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue
            if device is None:
                device = arg.device
            if device != arg.device:
                msg = 'Tensor on device ' + str(arg.device) + ' is not on the expected device ' + str(device) + '!'
                raise RuntimeError(msg)
        else:
            msg = 'Unexpected type when checking for same device, ' + str(type(arg)) + '!'
            raise RuntimeError(msg)