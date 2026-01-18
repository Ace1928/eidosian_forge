from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
@pytest.mark.parametrize('vectorize', [False, True])
def test_apply_gufunc_output_dtypes_string(vectorize):

    def stats(x):
        return np.mean(x, axis=-1)
    a = da.random.default_rng().normal(size=(10, 20, 30), chunks=(5, 5, 30))
    mean = apply_gufunc(stats, '(i)->()', a, output_dtypes='f', vectorize=vectorize)
    assert mean.compute().shape == (10, 20)