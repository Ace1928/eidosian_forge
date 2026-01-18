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
def test_fuse_subgraphs_linear_chains_of_duplicate_deps(compare_subgraph_callables):
    dsk = {'x-1': 1, 'add-1': (add, 'x-1', 'x-1'), 'add-2': (add, 'add-1', 'add-1'), 'add-3': (add, 'add-2', 'add-2'), 'add-4': (add, 'add-3', 'add-3'), 'add-5': (add, 'add-4', 'add-4')}
    res = fuse(dsk, 'add-5', fuse_subgraphs=True)
    sol = with_deps({'add-x-1': (SubgraphCallable({'x-1': 1, 'add-1': (add, 'x-1', 'x-1'), 'add-2': (add, 'add-1', 'add-1'), 'add-3': (add, 'add-2', 'add-2'), 'add-4': (add, 'add-3', 'add-3'), 'add-5': (add, 'add-4', 'add-4')}, 'add-5', ()),), 'add-5': 'add-x-1'})
    assert res == sol