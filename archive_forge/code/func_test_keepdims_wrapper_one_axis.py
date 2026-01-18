from __future__ import annotations
import pytest
import operator
import numpy as np
import dask.array as da
from dask.array.chunk import coarsen, getitem, keepdims_wrapper
def test_keepdims_wrapper_one_axis():

    def summer(a, axis=None):
        return a.sum(axis=axis)
    summer_wrapped = keepdims_wrapper(summer)
    assert summer_wrapped != summer
    a = np.arange(24).reshape(1, 2, 3, 4)
    r = summer(a, axis=2)
    rw = summer_wrapped(a, axis=2, keepdims=True)
    rwf = summer_wrapped(a, axis=2, keepdims=False)
    assert r.ndim == 3
    assert r.shape == (1, 2, 4)
    assert (r == np.array([[[12, 15, 18, 21], [48, 51, 54, 57]]])).all()
    assert rw.ndim == 4
    assert rw.shape == (1, 2, 1, 4)
    assert (rw == np.array([[[[12, 15, 18, 21]], [[48, 51, 54, 57]]]])).all()
    assert rwf.ndim == 3
    assert rwf.shape == (1, 2, 4)
    assert (rwf == np.array([[[12, 15, 18, 21], [48, 51, 54, 57]]])).all()