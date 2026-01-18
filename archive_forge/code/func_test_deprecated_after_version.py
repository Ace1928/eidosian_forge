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
def test_deprecated_after_version():

    @_deprecated(after_version='1.2.3')
    def foo():
        return 'bar'
    with pytest.warns(FutureWarning, match='deprecated after version 1.2.3'):
        assert foo() == 'bar'