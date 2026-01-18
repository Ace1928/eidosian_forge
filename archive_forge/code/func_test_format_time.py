from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def test_format_time():
    with pytest.warns(FutureWarning, match='dask.utils.format_time') as record:
        assert format_time(1.4) == ' 1.4s'
        assert format_time(10.4) == '10.4s'
        assert format_time(100.4) == ' 1min 40.4s'
        assert format_time(1000.4) == '16min 40.4s'
        assert format_time(10000.4) == ' 2hr 46min 40.4s'
    assert len(record) == 5