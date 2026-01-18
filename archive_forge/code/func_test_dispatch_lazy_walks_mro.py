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
def test_dispatch_lazy_walks_mro():
    """Check that subclasses of classes with lazily registered handlers still
    use their parent class's handler by default"""
    import decimal

    class Lazy(decimal.Decimal):
        pass

    class Eager(Lazy):
        pass
    foo = Dispatch()

    @foo.register(Eager)
    def eager_handler(x):
        return 'eager'

    def lazy_handler(a):
        return 'lazy'

    @foo.register_lazy('decimal')
    def register_decimal():
        foo.register(decimal.Decimal, lazy_handler)
    assert foo.dispatch(Lazy) == lazy_handler
    assert foo(Lazy(1)) == 'lazy'
    assert foo.dispatch(decimal.Decimal) == lazy_handler
    assert foo(decimal.Decimal(1)) == 'lazy'
    assert foo.dispatch(Eager) == eager_handler
    assert foo(Eager(1)) == 'eager'