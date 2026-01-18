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
def test_get_meta_library_gpu():
    cp = pytest.importorskip('cupy')
    cudf = pytest.importorskip('cudf')
    da = pytest.importorskip('dask.array')
    dd = pytest.importorskip('dask.dataframe')
    assert get_meta_library(cudf.DataFrame()) == cudf
    assert get_meta_library(cp.array([])) == cp
    assert get_meta_library(cudf.DataFrame()) == get_meta_library(cudf.DataFrame)
    assert get_meta_library(cp.ndarray([])) == get_meta_library(cp.ndarray)
    assert get_meta_library(cudf.DataFrame()) == get_meta_library(dd.from_dict({}, npartitions=1).to_backend('cudf'))
    assert get_meta_library(cp.ndarray([])) == get_meta_library(da.from_array([]).to_backend('cupy'))