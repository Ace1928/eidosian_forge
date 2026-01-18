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
def test_fuse_keys():
    fuse = fuse2
    d = {'a': 1, 'b': (inc, 'a'), 'c': (inc, 'b')}
    keys = ['b']
    assert fuse(d, keys, rename_keys=False) == with_deps({'b': (inc, 1), 'c': (inc, 'b')})
    assert fuse(d, keys, rename_keys=True) == with_deps({'a-b': (inc, 1), 'c': (inc, 'a-b'), 'b': 'a-b'})
    d = {'w': (inc, 'x'), 'x': (inc, 'y'), 'y': (inc, 'z'), 'z': (add, 'a', 'b'), 'a': 1, 'b': 2}
    keys = ['x', 'z']
    assert fuse(d, keys, rename_keys=False) == with_deps({'w': (inc, 'x'), 'x': (inc, (inc, 'z')), 'z': (add, 'a', 'b'), 'a': 1, 'b': 2})
    assert fuse(d, keys, rename_keys=True) == with_deps({'w': (inc, 'y-x'), 'y-x': (inc, (inc, 'z')), 'z': (add, 'a', 'b'), 'a': 1, 'b': 2, 'x': 'y-x'})