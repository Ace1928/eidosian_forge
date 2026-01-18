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
def fix_random_seed(seed: int=0) -> None:
    """Set the random seed across random, numpy.random and cupy.random."""
    random.seed(seed)
    numpy.random.seed(seed)
    if has_torch:
        torch.manual_seed(seed)
    if has_cupy_gpu:
        cupy.random.seed(seed)
        if has_torch and has_torch_cuda_gpu:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False