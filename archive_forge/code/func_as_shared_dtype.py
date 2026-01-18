from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
from importlib import import_module
import numpy as np
import pandas as pd
from numpy import all as array_all  # noqa
from numpy import any as array_any  # noqa
from numpy import (  # noqa
from numpy import concatenate as _concatenate
from numpy.lib.stride_tricks import sliding_window_view  # noqa
from packaging.version import Version
from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.options import OPTIONS
from xarray.core.utils import is_duck_array, is_duck_dask_array, module_available
from xarray.namedarray import pycompat
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, is_chunked_array
def as_shared_dtype(scalars_or_arrays, xp=np):
    """Cast a arrays to a shared dtype using xarray's type promotion rules."""
    array_type_cupy = array_type('cupy')
    if array_type_cupy and any((isinstance(x, array_type_cupy) for x in scalars_or_arrays)):
        import cupy as cp
        arrays = [asarray(x, xp=cp) for x in scalars_or_arrays]
    else:
        arrays = [asarray(x, xp=xp) for x in scalars_or_arrays]
    out_type = dtypes.result_type(*arrays)
    return [astype(x, out_type, copy=False) for x in arrays]