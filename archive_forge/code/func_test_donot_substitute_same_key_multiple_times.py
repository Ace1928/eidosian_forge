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
def test_donot_substitute_same_key_multiple_times():
    already_called = False

    def inc_only_once(x):
        nonlocal already_called
        if already_called:
            raise RuntimeError
        already_called = True
        return x + 1
    dsk = {'A': 42, 'B': (inc_only_once, 'A'), 'C': (add, 'B', 'B'), 'D': (inc, 'C')}
    dependencies = {'A': set(), 'B': {'A'}, 'C': {'B'}, 'D': {'C'}}
    fused_dsk = fuse(dsk, keys=['A', 'D'], dependencies=dependencies)[0]
    from dask.core import get
    assert get(fused_dsk, 'D') > 1