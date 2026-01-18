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
def test_accessors():
    x = np.random.default_rng().random((10, 10))
    dx = da.from_array(x, chunks=(3, 4))
    mx = np.ma.masked_greater(x, 0.4)
    dmx = da.ma.masked_greater(dx, 0.4)
    assert_eq(da.ma.getmaskarray(dmx), np.ma.getmaskarray(mx))
    assert_eq(da.ma.getmaskarray(dx), np.ma.getmaskarray(x))
    assert_eq(da.ma.getdata(dmx), np.ma.getdata(mx))
    assert_eq(da.ma.getdata(dx), np.ma.getdata(x))