import contextlib
import functools
import gc
import inspect
import logging
import multiprocessing
import os
import random
from statistics import mean
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed
def objects_are_equal(a: Any, b: Any, raise_exception: bool=False, dict_key: Optional[str]=None, rtol: Optional[float]=None, atol: Optional[float]=None) -> bool:
    """
    Test that two objects are equal. Tensors are compared to ensure matching
    size, dtype, device and values.
    """
    if type(a) is not type(b):
        if raise_exception:
            raise ValueError(f'type mismatch {type(a)} vs. {type(b)}')
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            if raise_exception:
                raise ValueError(f'keys mismatch {a.keys()} vs. {b.keys()}')
            return False
        for k in a.keys():
            if not objects_are_equal(a[k], b[k], raise_exception, k):
                return False
        return True
    elif isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            if raise_exception:
                raise ValueError(f'length mismatch {len(a)} vs. {len(b)}')
            return False
        return all((objects_are_equal(x, y, raise_exception) for x, y in zip(a, b)))
    elif torch.is_tensor(a):
        try:
            shape_dtype_device_match = a.size() == b.size() and a.dtype == b.dtype and (a.device == b.device)
            if not shape_dtype_device_match:
                if raise_exception:
                    msg = f'sizes: {a.size()} vs. {b.size()}, '
                    msg += f'types: {a.dtype} vs. {b.dtype}, '
                    msg += f'device: {a.device} vs. {b.device}'
                    raise AssertionError(msg)
                else:
                    return False
            if torch_version() < (1, 12, 0):
                torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
            else:
                torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            return True
        except (AssertionError, RuntimeError) as e:
            if raise_exception:
                if dict_key and isinstance(e, AssertionError):
                    msg = e.args[0]
                    new_msg = f"For dict key '{dict_key}': {msg}"
                    raise AssertionError(new_msg) from None
                else:
                    raise e
            else:
                return False
    else:
        return a == b