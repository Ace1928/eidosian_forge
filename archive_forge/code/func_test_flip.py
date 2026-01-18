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
@pytest.mark.parametrize('funcname, kwargs', [('flipud', {}), ('fliplr', {}), ('flip', {}), ('flip', {'axis': 0}), ('flip', {'axis': 1}), ('flip', {'axis': 2}), ('flip', {'axis': -1}), ('flip', {'axis': (0, 2)})])
@pytest.mark.parametrize('shape', [tuple(), (4,), (4, 6), (4, 6, 8), (4, 6, 8, 10)])
def test_flip(funcname, kwargs, shape):
    axis = kwargs.get('axis')
    if axis is None:
        if funcname == 'flipud':
            axis = (0,)
        elif funcname == 'fliplr':
            axis = (1,)
        elif funcname == 'flip':
            axis = range(len(shape))
    elif not isinstance(axis, tuple):
        axis = (axis,)
    np_a = np.random.default_rng().random(shape)
    da_a = da.from_array(np_a, chunks=1)
    np_func = getattr(np, funcname)
    da_func = getattr(da, funcname)
    try:
        for ax in axis:
            range(np_a.ndim)[ax]
    except IndexError:
        with pytest.raises(ValueError):
            da_func(da_a, **kwargs)
    else:
        np_r = np_func(np_a, **kwargs)
        da_r = da_func(da_a, **kwargs)
        assert_eq(np_r, da_r)