from __future__ import annotations
import contextlib
import functools
import itertools
import math
import numbers
import warnings
import numpy as np
from tlz import concat, frequencies
from dask.array.core import Array
from dask.array.numpy_compat import AxisError
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import has_keyword, is_arraylike, is_cupy_type, typename
def scipy_linalg_safe(func_name, *args, **kwargs):
    a = args[0]
    if is_cupy_type(a):
        import cupyx.scipy.linalg
        func = getattr(cupyx.scipy.linalg, func_name)
    else:
        import scipy.linalg
        func = getattr(scipy.linalg, func_name)
    return func(*args, **kwargs)