from __future__ import annotations
import pickle
import types
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from operator import add, setitem
from random import random
from typing import NamedTuple
import cloudpickle
import pytest
from tlz import merge
import dask
import dask.bag as db
from dask import compute
from dask.delayed import Delayed, delayed, to_task_dask
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import inc
@pytest.mark.filterwarnings('ignore:The dask.delayed:UserWarning')
def test_array_delayed():
    np = pytest.importorskip('numpy')
    da = pytest.importorskip('dask.array')
    arr = np.arange(100).reshape((10, 10))
    darr = da.from_array(arr, chunks=(5, 5))
    val = delayed(sum)([arr, darr, 1])
    assert isinstance(val, Delayed)
    assert np.allclose(val.compute(), arr + arr + 1)
    assert val.sum().compute() == (arr + arr + 1).sum()
    assert val[0, 0].compute() == (arr + arr + 1)[0, 0]
    task, dsk = to_task_dask(darr)
    assert not darr.dask.keys() - dsk.keys()
    diff = dsk.keys() - darr.dask.keys()
    assert len(diff) == 1
    delayed_arr = delayed(darr)
    assert (delayed_arr.compute() == arr).all()