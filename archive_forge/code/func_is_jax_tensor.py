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
def is_jax_tensor(x):
    """
    Tests if `x` is a Jax tensor or not. Safe to call even if jax is not installed.
    """
    return False if not is_flax_available() else _is_jax(x)