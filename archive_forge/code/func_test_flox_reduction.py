from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_flox_reduction(abcde):
    a, b, c, d, e = abcde
    g = 'g'
    dsk = {(a, 0): (f, 1), (a, 1, 0): (f, 1), (a, 1, 1): (f, 1), (a, 2, 0): (f, 1), (a, 2, 1): (f, 1), (b, 1, 0): (f, [(f, (a, 2, 0))]), (b, 1, 1): (f, [(f, (a, 2, 1))]), (b, 2, 1): (f, [(f, (a, 1, 1))]), (b, 2, 0): (f, [(f, (a, 1, 0))]), (b, 1, 1, 0): (b, 1, 0), (b, 1, 1, 1): (b, 1, 1), (b, 2, 2, 0): (b, 2, 0), (b, 2, 2, 1): (b, 2, 1), (c, 1, 0): (f, (b, 2, 2, 0)), (c, 1, 1): (f, (b, 2, 2, 1)), (c, 2, 0): (f, (b, 1, 1, 0)), (c, 2, 1): (f, (b, 1, 1, 1)), (d, 1): (f, [(f, (a, 1, 1), (a, 2, 1), (c, 1, 1), (c, 2, 1))]), (d, 0): (f, [(f, (a, 1, 0), (a, 2, 0), (c, 1, 0), (c, 2, 0))]), (e, 0): (d, 0), (e, 1): (d, 1), (g, 1, 0): (f, (a, 0), (b, 1, 1, 0)), (g, 1, 1): (f, (a, 0), (b, 2, 2, 0)), (g, 1, 2): (f, (a, 0), (e, 0)), (g, 2, 0): (f, (a, 0), (b, 1, 1, 1)), (g, 2, 1): (f, (a, 0), (b, 2, 2, 1)), (g, 2, 2): (f, (a, 0), (e, 1))}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    of1 = list((o[g, 1, ix] for ix in range(3)))
    of2 = list((o[g, 2, ix] for ix in range(3)))
    assert max(of1) < min(of2) or max(of2) < min(of1)