import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, ContextManager, Iterable, List, Tuple
import numpy as np
from packaging import version
from .import_utils import get_torch_version, is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy
def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return 'pt'
    elif representation.startswith("<class 'tensorflow."):
        return 'tf'
    elif representation.startswith("<class 'jax"):
        return 'jax'
    elif representation.startswith("<class 'numpy."):
        return 'np'