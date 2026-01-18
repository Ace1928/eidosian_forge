from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
@da.as_gufunc(signature='(n, n)->(n, n)', output_dtypes=float, vectorize=True)
def gufoo(x):
    return np.linalg.inv(x)