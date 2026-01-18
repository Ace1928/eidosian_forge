from __future__ import annotations
import itertools
import pickle
from functools import partial
import pytest
import dask
from dask.base import tokenize
from dask.core import get_dependencies
from dask.local import get_sync
from dask.optimization import (
from dask.utils import apply, partial_by_order
from dask.utils_test import add, inc
def test_inline_functions_non_hashable():

    class NonHashableCallable:

        def __call__(self, a):
            return a + 1

        def __hash__(self):
            raise TypeError('Not hashable')
    nohash = NonHashableCallable()
    dsk = {'a': 1, 'b': (inc, 'a'), 'c': (nohash, 'b'), 'd': (inc, 'c')}
    result = inline_functions(dsk, [], fast_functions={inc})
    assert result['c'] == (nohash, dsk['b'])
    assert 'b' not in result