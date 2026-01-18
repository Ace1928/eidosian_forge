from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('func1d_name, func1d, specify_output_props', [['ndim', lambda x: x.ndim, False], ['sum', lambda x: x.sum(), False], ['range', lambda x: [x.min(), x.max()], False], ['range2', lambda x: [[x.min(), x.max()], [x.max(), x.min()]], False], ['cumsum', lambda x: np.cumsum(x), True]])
@pytest.mark.parametrize('input_shape, axis', [[(10, 15, 20), 0], [(10, 15, 20), 1], [(10, 15, 20), 2], [(10, 15, 20), -1]])
def test_apply_along_axis(func1d_name, func1d, specify_output_props, input_shape, axis):
    a = np.random.default_rng().integers(0, 10, input_shape)
    d = da.from_array(a, chunks=len(input_shape) * (5,))
    output_shape = None
    output_dtype = None
    if specify_output_props:
        slices = [0] * a.ndim
        slices[axis] = slice(None)
        slices = tuple(slices)
        sample = np.array(func1d(a[slices]))
        output_shape = sample.shape
        output_dtype = sample.dtype
    assert_eq(da.apply_along_axis(func1d, axis, d, dtype=output_dtype, shape=output_shape), np.apply_along_axis(func1d, axis, a))