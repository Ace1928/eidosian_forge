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
def mxnet2xp(mx_tensor: 'mx.nd.NDArray', *, ops: Optional['Ops']=None) -> ArrayXd:
    """Convert a MXNet tensor to a numpy or cupy tensor."""
    from .api import NumpyOps
    assert_mxnet_installed()
    if is_mxnet_gpu_array(mx_tensor):
        if isinstance(ops, NumpyOps):
            return mx_tensor.detach().asnumpy()
        else:
            return cupy_from_dlpack(mx_tensor.to_dlpack_for_write())
    elif isinstance(ops, NumpyOps) or ops is None:
        return mx_tensor.detach().asnumpy()
    else:
        return cupy.asarray(mx_tensor.asnumpy())