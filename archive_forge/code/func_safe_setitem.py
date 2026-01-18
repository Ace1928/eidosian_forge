from __future__ import annotations
import warnings
from collections.abc import Hashable, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Union
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def safe_setitem(dest, key: Hashable, value, name: T_Name=None):
    if key in dest:
        var_str = f' on variable {name!r}' if name else ''
        raise ValueError(f"failed to prevent overwriting existing key {key} in attrs{var_str}. This is probably an encoding field used by xarray to describe how a variable is serialized. To proceed, remove this key from the variable's attributes manually.")
    dest[key] = value