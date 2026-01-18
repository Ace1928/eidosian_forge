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
def assert_tensorflow_installed() -> None:
    """Raise an ImportError if TensorFlow is not installed."""
    template = 'TensorFlow support requires {pkg}: pip install thinc[tensorflow]\n\nEnable TensorFlow support with thinc.api.enable_tensorflow()'
    if not has_tensorflow:
        raise ImportError(template.format(pkg='tensorflow>=2.0.0,<2.6.0'))