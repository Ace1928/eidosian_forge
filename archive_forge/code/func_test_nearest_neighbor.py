from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_nearest_neighbor(abcde):
    """

    a1  a2  a3  a4  a5  a6  a7 a8  a9
     \\  |  /  \\ |  /  \\ |  / \\ |  /
        b1      b2      b3     b4

    Want to finish off a local group before moving on.
    This is difficult because all groups are connected.
    """
    a, b, c, _, _ = abcde
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = (a + i for i in '123456789')
    b1, b2, b3, b4 = (b + i for i in '1234')
    dsk = {b1: (f,), b2: (f,), b3: (f,), b4: (f,), a1: (f, b1), a2: (f, b1), a3: (f, b1, b2), a4: (f, b2), a5: (f, b2, b3), a6: (f, b3), a7: (f, b3, b4), a8: (f, b4), a9: (f, b4)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert 3 < sum((o[a + i] < len(o) / 2 for i in '123456789')) < 7
    assert 1 < sum((o[b + i] < len(o) / 2 for i in '1234')) < 4