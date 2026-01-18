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
def test_cached_cumsum_nan():
    np = pytest.importorskip('numpy')
    a = (1, np.nan, 3)
    x = cached_cumsum(a)
    y = cached_cumsum(a, initial_zero=True)
    np.testing.assert_equal(x, (1, np.nan, np.nan))
    np.testing.assert_equal(y, (0, 1, np.nan, np.nan))