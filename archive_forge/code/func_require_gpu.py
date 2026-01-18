import contextlib
import functools
import inspect
import os
import platform
import random
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
import numpy
from packaging.version import Version
from wasabi import table
from .compat import (
from .compat import mxnet as mx
from .compat import tensorflow as tf
from .compat import torch
from typing import TYPE_CHECKING
from . import types  # noqa: E402
from .types import ArgsKwargs, ArrayXd, FloatsXd, IntsXd, Padded, Ragged  # noqa: E402
def require_gpu(gpu_id: int=0) -> bool:
    from .backends import CupyOps, MPSOps, set_current_ops
    if platform.system() == 'Darwin' and (not has_torch_mps):
        if has_torch:
            raise ValueError('Cannot use GPU, installed PyTorch does not support MPS')
        raise ValueError('Cannot use GPU, PyTorch is not installed')
    elif platform.system() != 'Darwin' and (not has_cupy):
        raise ValueError('Cannot use GPU, CuPy is not installed')
    elif not has_gpu:
        raise ValueError('No GPU devices detected')
    if has_cupy_gpu:
        set_current_ops(CupyOps())
        set_active_gpu(gpu_id)
    else:
        set_current_ops(MPSOps())
    return True