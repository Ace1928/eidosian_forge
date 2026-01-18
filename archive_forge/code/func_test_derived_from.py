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
def test_derived_from():

    class Foo:

        def f(a, b):
            """A super docstring

            An explanation

            Parameters
            ----------
            a: int
                an explanation of a
            b: float
                an explanation of b
            """

    class Bar:

        @derived_from(Foo)
        def f(a, c):
            pass

    class Zap:

        @derived_from(Foo)
        def f(a, c):
            """extra docstring"""
            pass
    assert Bar.f.__doc__.strip().startswith('A super docstring')
    assert 'Foo.f' in Bar.f.__doc__
    assert any(('inconsistencies' in line for line in Bar.f.__doc__.split('\n')[:7]))
    [b_arg] = [line for line in Bar.f.__doc__.split('\n') if 'b:' in line]
    assert 'not supported' in b_arg.lower()
    assert 'dask' in b_arg.lower()
    assert '  extra docstring\n\n' in Zap.f.__doc__