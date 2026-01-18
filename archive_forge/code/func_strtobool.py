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
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {'y', 'yes', 't', 'true', 'on', '1'}:
        return 1
    if val in {'n', 'no', 'f', 'false', 'off', '0'}:
        return 0
    raise ValueError(f'invalid truth value {val!r}')