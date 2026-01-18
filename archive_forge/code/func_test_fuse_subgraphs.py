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
def test_fuse_subgraphs(compare_subgraph_callables):
    dsk = {'x-1': 1, 'inc-1': (inc, 'x-1'), 'inc-2': (inc, 'inc-1'), 'add-1': (add, 'x-1', 'inc-2'), 'inc-3': (inc, 'add-1'), 'inc-4': (inc, 'inc-3'), 'add-2': (add, 'add-1', 'inc-4'), 'inc-5': (inc, 'add-2'), 'inc-6': (inc, 'inc-5')}
    res = fuse(dsk, 'inc-6', fuse_subgraphs=True)
    sol = with_deps({'inc-6': 'add-inc-x-1', 'add-inc-x-1': (SubgraphCallable({'x-1': 1, 'add-1': (add, 'x-1', (inc, (inc, 'x-1'))), 'inc-6': (inc, (inc, (add, 'add-1', (inc, (inc, 'add-1')))))}, 'inc-6', ()),)})
    assert res == sol
    res = fuse(dsk, 'inc-6', fuse_subgraphs=True, rename_keys=False)
    sol = with_deps({'inc-6': (SubgraphCallable({'x-1': 1, 'add-1': (add, 'x-1', (inc, (inc, 'x-1'))), 'inc-6': (inc, (inc, (add, 'add-1', (inc, (inc, 'add-1')))))}, 'inc-6', ()),)})
    assert res == sol
    res = fuse(dsk, 'add-2', fuse_subgraphs=True)
    sol = with_deps({'add-inc-x-1': (SubgraphCallable({'x-1': 1, 'add-1': (add, 'x-1', (inc, (inc, 'x-1'))), 'add-2': (add, 'add-1', (inc, (inc, 'add-1')))}, 'add-2', ()),), 'add-2': 'add-inc-x-1', 'inc-6': (inc, (inc, 'add-2'))})
    assert res == sol
    res = fuse(dsk, 'inc-2', fuse_subgraphs=True)
    sols = []
    for inkeys in itertools.permutations(('x-1', 'inc-2')):
        sols.append(with_deps({'x-1': 1, 'inc-2': (inc, (inc, 'x-1')), 'inc-6': 'inc-add-1', 'inc-add-1': (SubgraphCallable({'add-1': (add, 'x-1', 'inc-2'), 'inc-6': (inc, (inc, (add, 'add-1', (inc, (inc, 'add-1')))))}, 'inc-6', inkeys),) + inkeys}))
    assert res in sols
    res = fuse(dsk, ['inc-2', 'add-2'], fuse_subgraphs=True)
    sols = []
    for inkeys in itertools.permutations(('x-1', 'inc-2')):
        sols.append(with_deps({'x-1': 1, 'inc-2': (inc, (inc, 'x-1')), 'inc-add-1': (SubgraphCallable({'add-1': (add, 'x-1', 'inc-2'), 'add-2': (add, 'add-1', (inc, (inc, 'add-1')))}, 'add-2', inkeys),) + inkeys, 'add-2': 'inc-add-1', 'inc-6': (inc, (inc, 'add-2'))}))
    assert res in sols