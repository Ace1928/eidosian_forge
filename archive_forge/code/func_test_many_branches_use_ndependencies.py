from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_many_branches_use_ndependencies(abcde):
    """From https://github.com/dask/dask/pull/5646#issuecomment-562700533

    Sometimes we need larger or wider DAGs to test behavior.  This test
    ensures we choose the branch with more work twice in succession.
    This is important, because ``order`` may search along dependencies
    and then along dependents.

    """
    a, b, c, d, e = abcde
    dd = d + d
    ee = e + e
    dsk = {(a, 0): 0, (a, 1): (f, (a, 0)), (a, 2): (f, (a, 1)), (b, 1): (f, (a, 0)), (b, 2): (f, (b, 1)), (c, 1): (f, (a, 0)), (d, 1): (f, (a, 0)), (d, 2): (f, (d, 1)), (dd, 1): (f, (a, 0)), (dd, 2): (f, (dd, 1)), (dd, 3): (f, (d, 2), (dd, 2)), (e, 1): (f, (a, 0)), (e, 2): (f, (e, 1)), (ee, 1): (f, (a, 0)), (ee, 2): (f, (ee, 1)), (ee, 3): (f, (e, 2), (ee, 2)), (a, 3): (f, (a, 2), (b, 2), (c, 1), (dd, 3), (ee, 3))}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    actual = sorted((v for (letter, _), v in o.items() if letter in {d, dd, e, ee}))
    assert actual == expected
    assert o[c, 1] == o[a, 3] - 1