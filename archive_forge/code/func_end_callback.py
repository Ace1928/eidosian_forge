from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def end_callback(key, value, d, state, worker_id):
    assert key == 'a' or key is None
    assert value == 2 or value is None
    assert d == dsk
    assert isinstance(state, dict)