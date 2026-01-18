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
def test_ensure_dict():
    d = {'x': 1}
    assert ensure_dict(d) is d

    class mydict(dict):
        pass
    d2 = ensure_dict(d, copy=True)
    d3 = ensure_dict(HighLevelGraph.from_collections('x', d))
    d4 = ensure_dict(mydict(d))
    for di in (d2, d3, d4):
        assert type(di) is dict
        assert di is not d
        assert di == d