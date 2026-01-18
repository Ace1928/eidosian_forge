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
def prefer_gpu(gpu_id: int=0) -> bool:
    """Use GPU if it's available. Returns True if so, False otherwise."""
    if has_gpu:
        require_gpu(gpu_id=gpu_id)
    return has_gpu