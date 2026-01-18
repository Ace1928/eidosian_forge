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
@pytest.mark.parametrize('keepdims', [False, True])
def test_average_weights_with_masked_array(keepdims):
    mask = np.array([[True, False], [True, True], [False, True]])
    data = np.arange(6).reshape((3, 2))
    a = np.ma.array(data, mask=mask)
    d_a = da.ma.masked_array(data=data, mask=mask, chunks=2)
    weights = np.array([0.25, 0.75])
    d_weights = da.from_array(weights, chunks=2)
    da_avg = da.ma.average(d_a, weights=d_weights, axis=1, keepdims=keepdims)
    if NUMPY_GE_123:
        assert_eq(da_avg, np.ma.average(a, weights=weights, axis=1, keepdims=keepdims))
    elif not keepdims:
        assert_eq(da_avg, np.ma.average(a, weights=weights, axis=1))