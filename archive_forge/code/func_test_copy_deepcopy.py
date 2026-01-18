from __future__ import annotations
import random
import sys
from copy import deepcopy
from itertools import product
import numpy as np
import pytest
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, ComplexWarning
from dask.array.utils import assert_eq
from dask.base import tokenize
from dask.utils import typename
def test_copy_deepcopy():
    t = np.ma.masked_array([1, 2], mask=[0, 1])
    x = da.from_array(t, chunks=t.shape, asarray=False)
    y = x.copy()
    memo = {}
    y2 = deepcopy(x, memo=memo)
    xx = da.ma.masked_where([False, True], [1, 2])
    assert_eq(x, xx)
    assert_eq(y, t)
    assert isinstance(y.compute(), np.ma.masked_array)
    assert_eq(y2, t)
    assert isinstance(y2.compute(), np.ma.masked_array)