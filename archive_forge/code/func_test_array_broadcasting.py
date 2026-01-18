from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
@pytest.mark.skipif(not hasattr(np, 'broadcast_to'), reason='requires numpy 1.10 method "broadcast_to"')
def test_array_broadcasting(generator_class):
    arr = np.arange(6).reshape((2, 3))
    daones = da.ones((2, 3, 4), chunks=3)
    assert generator_class().poisson(arr, chunks=3).compute().shape == (2, 3)
    for x in (arr, daones):
        y = generator_class().normal(x, 2, chunks=3)
        assert y.shape == x.shape
        assert y.compute().shape == x.shape
    y = generator_class().normal(daones, 2, chunks=3)
    assert set(daones.dask).issubset(set(y.dask))
    assert generator_class().normal(np.ones((1, 4)), da.ones((2, 3, 4), chunks=(2, 3, 4)), chunks=(2, 3, 4)).compute().shape == (2, 3, 4)
    assert generator_class().normal(scale=np.ones((1, 4)), loc=da.ones((2, 3, 4), chunks=(2, 3, 4)), size=(2, 2, 3, 4), chunks=(2, 2, 3, 4)).compute().shape == (2, 2, 3, 4)
    with pytest.raises(ValueError):
        generator_class().normal(arr, np.ones((3, 1)), size=(2, 3, 4), chunks=3)
    for o in (np.ones(100), da.ones(100, chunks=(50,)), 1):
        a = generator_class().normal(1000 * o, 0.01, chunks=(50,))
        assert 800 < a.mean().compute() < 1200
    x = np.arange(10) ** 3
    y = da.from_array(x, chunks=(1,))
    z = generator_class().normal(y, 0.01, chunks=(10,))
    assert 0.8 < z.mean().compute() / x.mean() < 1.2