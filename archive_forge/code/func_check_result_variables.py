from __future__ import annotations
import collections
import itertools
import operator
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict
import numpy as np
from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index
from xarray.core.merge import merge
from xarray.core.utils import is_dask_collection
from xarray.core.variable import Variable
def check_result_variables(result: DataArray | Dataset, expected: ExpectedDict, kind: Literal['coords', 'data_vars']):
    if kind == 'coords':
        nice_str = 'coordinate'
    elif kind == 'data_vars':
        nice_str = 'data'
    missing = expected[kind] - set(getattr(result, kind))
    if missing:
        raise ValueError(f'Result from applying user function does not contain {nice_str} variables {missing}.')
    extra = set(getattr(result, kind)) - expected[kind]
    if extra:
        raise ValueError(f'Result from applying user function has unexpected {nice_str} variables {extra}.')