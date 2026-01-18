from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_method_caller():
    a = [1, 2, 3, 3, 3]
    f = methodcaller('count')
    assert f(a, 3) == a.count(3)
    assert methodcaller('count') is f
    assert M.count is f
    assert pickle.loads(pickle.dumps(f)) is f
    assert 'count' in dir(M)
    assert 'count' in str(methodcaller('count'))
    assert 'count' in repr(methodcaller('count'))