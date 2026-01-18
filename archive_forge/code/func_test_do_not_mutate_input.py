from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_do_not_mutate_input():
    np = pytest.importorskip('numpy')
    dsk = {'a': 'data', 'b': (f, 1), 'c': np.array([[1, 2], [3, 4]]), 'd': ['a', 'b', 'c'], 'e': (f, 'd')}
    dependencies, __build_class__ = get_deps(dsk)
    dependencies_copy = dependencies.copy()
    dsk_copy = dsk.copy()
    o = order(dsk, dependencies=dependencies)
    assert_topological_sort(dsk, o)
    assert dsk == dsk_copy
    assert dependencies == dependencies_copy