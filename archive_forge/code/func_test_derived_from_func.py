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
def test_derived_from_func():
    import builtins

    @derived_from(builtins)
    def sum():
        """extra docstring"""
        pass
    assert 'extra docstring\n\n' in sum.__doc__
    assert 'Return the sum of' in sum.__doc__
    assert 'This docstring was copied from builtins.sum' in sum.__doc__