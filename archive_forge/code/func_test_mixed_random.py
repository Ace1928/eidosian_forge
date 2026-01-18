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
@pytest.mark.parametrize('func', functions)
@pytest.mark.filterwarnings(f'ignore::{typename(ComplexWarning)}')
def test_mixed_random(func):
    d = da.random.default_rng().random((4, 3, 4), chunks=(1, 2, 2))
    d[d < 0.4] = 0
    fn = lambda x: np.ma.masked_equal(x, 0) if random.random() < 0.5 else x
    s = d.map_blocks(fn)
    dd = func(d)
    ss = func(s)
    assert_eq(dd, ss, check_meta=False, check_type=False)