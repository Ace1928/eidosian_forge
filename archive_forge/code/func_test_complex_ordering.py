from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_complex_ordering():
    da = pytest.importorskip('dask.array')
    from dask.diagnostics import Callback
    actual_order = []

    def track_order(key, dask, state):
        actual_order.append(key)
    x = da.random.normal(size=(20, 20), chunks=(-1, -1))
    res = (x.dot(x.T) - x.mean(axis=0)).std()
    dsk = dict(res.__dask_graph__())
    exp_order_dict = order(dsk)
    exp_order = sorted(exp_order_dict.keys(), key=exp_order_dict.get)
    with Callback(pretask=track_order):
        get_sync(dsk, exp_order[-1])
    assert actual_order == exp_order