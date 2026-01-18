from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
@classmethod
def from_variables(cls, variables: Mapping[Any, Variable], *, options: Mapping[str, Any]) -> PandasMultiIndex:
    _check_dim_compat(variables)
    dim = next(iter(variables.values())).dims[0]
    index = pd.MultiIndex.from_arrays([var.values for var in variables.values()], names=variables.keys())
    index.name = dim
    level_coords_dtype = {name: var.dtype for name, var in variables.items()}
    obj = cls(index, dim, level_coords_dtype=level_coords_dtype)
    return obj