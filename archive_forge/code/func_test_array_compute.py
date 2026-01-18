from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def test_array_compute(capsys):
    da = pytest.importorskip('dask.array')
    data = da.ones((100, 100), dtype='f4', chunks=(100, 100))
    with ProgressBar():
        out = data.sum().compute()
    assert out == 10000
    check_bar_completed(capsys)