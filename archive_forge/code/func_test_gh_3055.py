from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_gh_3055():
    da = pytest.importorskip('dask.array')
    A, B = (20, 99)
    orig = x = da.random.normal(size=(A, B), chunks=(1, None))
    for _ in range(2):
        y = (x[:, None, :] * x[:, :, None]).cumsum(axis=0)
        x = x.cumsum(axis=0)
    w = (y * x[:, None]).sum(axis=(1, 2))
    dsk = dict(w.__dask_graph__())
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert max(diagnostics(dsk, o=o)[1]) <= 8
    L = [o[k] for k in w.__dask_keys__()]
    assert sum((x < len(o) / 2 for x in L)) > len(L) / 3
    L = [o[k] for kk in orig.__dask_keys__() for k in kk]
    assert sum((x > len(o) / 2 for x in L)) > len(L) / 3
    assert sorted(L) == L