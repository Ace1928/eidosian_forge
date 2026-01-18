from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, Union
import numpy as np
import pandas as pd
from xarray.coding import strings, times, variables
from xarray.coding.variables import SerializationWarning, pop_to
from xarray.core import indexing
from xarray.core.common import (
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import IndexVariable, Variable
from xarray.namedarray.utils import is_duck_dask_array
def ensure_not_multiindex(var: Variable, name: T_Name=None) -> None:
    if isinstance(var._data, indexing.PandasMultiIndexingAdapter):
        if name is None and isinstance(var, IndexVariable):
            name = var.name
        if var.dims == (name,):
            raise NotImplementedError(f'variable {name!r} is a MultiIndex, which cannot yet be serialized. Instead, either use reset_index() to convert MultiIndex levels into coordinate variables instead or use https://cf-xarray.readthedocs.io/en/latest/coding.html.')