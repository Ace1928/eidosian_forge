from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_avoid_upwards_branching_complex(abcde):
    """
         a1
         |
    e2   a2  d2  d3
    |    |    \\  /
    e1   a3    d1
     \\  /  \\  /
      b1    c1
      |     |
      b2    c2
            |
            c3

    Prefer c1 over b1 because c1 will stay in memory less long while b1
    computes
    """
    a, b, c, d, e = abcde
    dsk = {(a, 1): (f, (a, 2)), (a, 2): (f, (a, 3)), (a, 3): (f, (b, 1), (c, 1)), (b, 1): (f, (b, 2)), (b, 2): (f,), (c, 1): (f, (c, 2)), (c, 2): (f, (c, 3)), (c, 3): (f,), (d, 1): (f, (c, 1)), (d, 2): (f, (d, 1)), (d, 3): (f, (d, 1)), (e, 1): (f, (b, 1)), (e, 2): (f, (e, 1))}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o[c, 1] < o[b, 1]
    assert abs(o[d, 2] - o[d, 3]) == 1