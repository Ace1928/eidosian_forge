from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_infer_dtype():
    x = np.arange(50).reshape((5, 10))
    y = np.arange(10)
    dx = da.from_array(x, chunks=5)
    dy = da.from_array(y, chunks=5)

    def foo(x, *args, **kwargs):
        cast = kwargs.pop('cast', 'i8')
        return (x + sum(args)).astype(cast)
    dz = apply_gufunc(foo, '(),(),()->()', dx, dy, 1)
    z = foo(dx, dy, 1)
    assert_eq(dz, z)
    dz = apply_gufunc(foo, '(),(),()->()', dx, dy, 1, cast='f8')
    z = foo(dx, dy, 1, cast='f8')
    assert_eq(dz, z)
    dz = apply_gufunc(foo, '(),(),()->()', dx, dy, 1, cast='f8', output_dtypes='f8')
    z = foo(dx, dy, 1, cast='f8')
    assert_eq(dz, z)

    def foo(x):
        raise RuntimeError('Woops')
    with pytest.raises(ValueError) as e:
        apply_gufunc(foo, '()->()', dx)
    msg = str(e.value)
    assert msg.startswith('`dtype` inference failed')
    assert 'Please specify the dtype explicitly' in msg
    assert 'RuntimeError' in msg

    def foo(x, y):
        return (x + y, x - y)
    z0, z1 = apply_gufunc(foo, '(),()->(),()', dx, dy)
    assert_eq(z0, dx + dy)
    assert_eq(z1, dx - dy)