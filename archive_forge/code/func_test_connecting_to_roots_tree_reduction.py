from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_connecting_to_roots_tree_reduction():
    dsk = {'a0': (f, 1), 'a1': (f, 1), 'a2': (f, 1), 'a3': (f, 1), 'b1': (f, 'a0'), 'b2': (f, 'a1'), 'b3': (f, 'a2'), 'b4': (f, 'a3'), 'c1': (f, 'b1', 'b2'), 'c2': (f, 'b3', 'b4'), 'd': (f, 'c1', 'c2')}
    dependencies, dependents = get_deps(dsk)
    connected_roots, max_dependents = _connecting_to_roots(dependencies, dependents)
    assert connected_roots == {'a0': set(), 'a1': set(), 'a2': set(), 'a3': set(), 'b1': {'a0'}, 'b2': {'a1'}, 'b3': {'a2'}, 'b4': {'a3'}, 'c1': {'a1', 'a0'}, 'c2': {'a3', 'a2'}, 'd': {'a1', 'a2', 'a3', 'a0'}}
    assert all((v == 1 for v in max_dependents.values()))
    connected_roots, max_dependents = _connecting_to_roots(dependents, dependencies)
    assert connected_roots.pop('d') == set()
    assert all((v == {'d'} for v in connected_roots.values())), set(connected_roots.values())
    assert all((v == 2 for v in max_dependents.values()))