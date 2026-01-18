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
def test_ensure_bytes_ndarray():
    np = pytest.importorskip('numpy')
    result = ensure_bytes(np.arange(12))
    assert isinstance(result, bytes)