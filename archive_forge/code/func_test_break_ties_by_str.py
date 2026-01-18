from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_break_ties_by_str(abcde):
    a, b, c, d, e = abcde
    dsk = {('x', i): (inc, i) for i in range(10)}
    x_keys = sorted(dsk)
    dsk['y'] = (f, list(x_keys))
    o = order(dsk)
    assert_topological_sort(dsk, o)
    expected = {'y': 10}
    expected.update({k: i for i, k in enumerate(x_keys[::-1])})
    assert o == expected