from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
def test_custom_collection():
    dsk = {('x', h1): 1, ('x', h2): 2}
    dsk2 = {('y', h1): (add, ('x', h1), ('x', h2)), ('y', h2): (add, ('y', h1), 1)}
    dsk2.update(dsk)
    dsk3 = {'z': (add, ('y', h1), ('y', h2))}
    dsk3.update(dsk2)
    w = Tuple({}, [])
    x = Tuple(dsk, [('x', h1), ('x', h2)])
    y = Tuple(dsk2, [('y', h1), ('y', h2)])
    z = Tuple(dsk3, ['z'])
    t = w + x + y + z
    with pytest.raises(AttributeError):
        x.foo = 1
    assert is_dask_collection(w)
    assert is_dask_collection(x)
    assert is_dask_collection(y)
    assert is_dask_collection(z)
    assert is_dask_collection(t)
    assert tokenize(w) == tokenize(w)
    assert tokenize(x) == tokenize(x)
    assert tokenize(y) == tokenize(y)
    assert tokenize(z) == tokenize(z)
    assert tokenize(t) == tokenize(t)
    assert len({tokenize(coll) for coll in (w, x, y, z, t)}) == 5
    assert get_collection_names(w) == set()
    assert get_collection_names(x) == {'x'}
    assert get_collection_names(y) == {'y'}
    assert get_collection_names(z) == {'z'}
    assert get_collection_names(t) == {'x', 'y', 'z'}
    assert w.compute() == ()
    assert x.compute() == (1, 2)
    assert y.compute() == (3, 4)
    assert z.compute() == (7,)
    assert dask.compute(w, [{'x': x}, y, z]) == ((), [{'x': (1, 2)}, (3, 4), (7,)])
    assert t.compute() == (1, 2, 3, 4, 7)
    t2 = t.persist()
    assert isinstance(t2, Tuple)
    assert t2._keys == t._keys
    assert sorted(t2._dask.values()) == [1, 2, 3, 4, 7]
    assert t2.compute() == (1, 2, 3, 4, 7)
    w2, x2, y2, z2 = dask.persist(w, x, y, z)
    assert y2._keys == y._keys
    assert y2._dask == {('y', h1): 3, ('y', h2): 4}
    assert y2.compute() == (3, 4)
    t3 = x2 + y2 + z2
    assert t3.compute() == (1, 2, 3, 4, 7)
    rebuild, args = w.__dask_postpersist__()
    w3 = rebuild({}, *args, rename={'w': 'w3'})
    assert w3.compute() == ()
    rebuild, args = x.__dask_postpersist__()
    x3 = rebuild({('x3', h1): 10, ('x3', h2): 20}, *args, rename={'x': 'x3'})
    assert x3.compute() == (10, 20)
    rebuild, args = z.__dask_postpersist__()
    z3 = rebuild({'z3': 70}, *args, rename={'z': 'z3'})
    assert z3.compute() == (70,)