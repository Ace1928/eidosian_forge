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
def test_derived_from_dask_dataframe():
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled():
        pytest.xfail("we don't have docs yet")
    assert 'inconsistencies' in dd.DataFrame.dropna.__doc__
    [axis_arg] = [line for line in dd.DataFrame.dropna.__doc__.split('\n') if 'axis :' in line]
    assert 'not supported' in axis_arg.lower()
    assert 'dask' in axis_arg.lower()
    assert 'Object with missing values filled' in dd.DataFrame.ffill.__doc__