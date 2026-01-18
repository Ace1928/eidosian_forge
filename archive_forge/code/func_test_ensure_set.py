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
def test_ensure_set():
    s = {1}
    assert ensure_set(s) is s

    class myset(set):
        pass
    s2 = ensure_set(s, copy=True)
    s3 = ensure_set(myset(s))
    for si in (s2, s3):
        assert type(si) is set
        assert si is not s
        assert si == s