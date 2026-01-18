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
def infer_template(func: Callable[..., T_Xarray], obj: DataArray | Dataset, *args, **kwargs) -> T_Xarray:
    """Infer return object by running the function on meta objects."""
    meta_args = [make_meta(arg) for arg in (obj,) + args]
    try:
        template = func(*meta_args, **kwargs)
    except Exception as e:
        raise Exception("Cannot infer object returned from running user provided function. Please supply the 'template' kwarg to map_blocks.") from e
    if not isinstance(template, (Dataset, DataArray)):
        raise TypeError(f'Function must return an xarray DataArray or Dataset. Instead it returned {type(template)}')
    return template