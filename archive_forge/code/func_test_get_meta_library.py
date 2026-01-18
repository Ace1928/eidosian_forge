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
def test_get_meta_library():
    np = pytest.importorskip('numpy')
    pd = pytest.importorskip('pandas')
    da = pytest.importorskip('dask.array')
    dd = pytest.importorskip('dask.dataframe')
    assert get_meta_library(pd.DataFrame()) == pd
    assert get_meta_library(np.array([])) == np
    assert get_meta_library(pd.DataFrame()) == get_meta_library(pd.DataFrame)
    assert get_meta_library(np.ndarray([])) == get_meta_library(np.ndarray)
    assert get_meta_library(pd.DataFrame()) == get_meta_library(dd.from_dict({}, npartitions=1))
    assert get_meta_library(np.ndarray([])) == get_meta_library(da.from_array([]))