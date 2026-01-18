from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_avoid_upwards_branching(abcde):
    """
       a1
       |
       a2
       |
       a3    d1
      /  \\  /
    b1    c1
    |     |
    b2    c2
          |
          c3
    """
    a, b, c, d, e = abcde
    dsk = {(a, 1): (f, (a, 2)), (a, 2): (f, (a, 3)), (a, 3): (f, (b, 1), (c, 1)), (b, 1): (f, (b, 2)), (c, 1): (f, (c, 2)), (c, 2): (f, (c, 3)), (d, 1): (f, (c, 1)), (c, 3): 1, (b, 2): 1}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o[d, 1] < o[b, 1]