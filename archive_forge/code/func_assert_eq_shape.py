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
def assert_eq_shape(a, b, check_ndim=True, check_nan=True):
    if check_ndim:
        assert len(a) == len(b)
    for aa, bb in zip(a, b):
        if math.isnan(aa) or math.isnan(bb):
            if check_nan:
                assert math.isnan(aa) == math.isnan(bb)
        else:
            assert aa == bb