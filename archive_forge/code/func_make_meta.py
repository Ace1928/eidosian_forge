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
def make_meta(obj):
    """If obj is a DataArray or Dataset, return a new object of the same type and with
    the same variables and dtypes, but where all variables have size 0 and numpy
    backend.
    If obj is neither a DataArray nor Dataset, return it unaltered.
    """
    if isinstance(obj, DataArray):
        obj_array = obj
        obj = dataarray_to_dataset(obj)
    elif isinstance(obj, Dataset):
        obj_array = None
    else:
        return obj
    from dask.array.utils import meta_from_array
    meta = Dataset()
    for name, variable in obj.variables.items():
        meta_obj = meta_from_array(variable.data, ndim=variable.ndim)
        meta[name] = (variable.dims, meta_obj, variable.attrs)
    meta.attrs = obj.attrs
    meta = meta.set_coords(obj.coords)
    if obj_array is not None:
        return dataset_to_dataarray(meta)
    return meta