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
def test_dispatch_variadic_on_first_argument():
    foo = Dispatch()
    foo.register(int, lambda a, b: a + b)
    foo.register(float, lambda a, b: a - b)
    assert foo(1, 2) == 3
    assert foo(1.0, 2.0) == -1