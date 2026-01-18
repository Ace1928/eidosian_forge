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
def test_SubgraphCallable():
    non_hashable = [1, 2, 3]
    dsk = {'a': (apply, add, ['in1', 2]), 'b': (apply, partial_by_order, ['in2'], {'function': func_with_kwargs, 'other': [(1, 20)], 'c': 4}), 'c': (apply, partial_by_order, ['in2', 'in1'], {'function': func_with_kwargs, 'other': [(1, 20)]}), 'd': (inc, 'a'), 'e': (add, 'c', 'd'), 'f': ['a', 2, 'b', (add, 'b', (sum, non_hashable))], 'h': (add, (sum, 'f'), (sum, ['a', 'b']))}
    f = SubgraphCallable(dsk, 'h', ['in1', 'in2'], name='test')
    assert f.name == 'test'
    assert repr(f) == 'test'
    f2 = SubgraphCallable(dsk, 'h', ['in1', 'in2'], name='test')
    assert f == f2
    f3 = SubgraphCallable(dsk, 'g', ['in1', 'in2'], name='test')
    assert f != f3
    assert hash(SubgraphCallable(None, None, [None]))
    assert hash(f3) != hash(f2)
    dsk2 = dsk.copy()
    dsk2.update({'in1': 1, 'in2': 2})
    assert f(1, 2) == get_sync(cull(dsk2, ['h'])[0], ['h'])[0]
    assert f(1, 2) == f(1, 2)
    f2 = pickle.loads(pickle.dumps(f))
    assert f2 == f
    assert hash(f2) == hash(f)
    assert f2(1, 2) == f(1, 2)