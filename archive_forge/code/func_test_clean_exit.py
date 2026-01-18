from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
@pytest.mark.parametrize('get', [get_threaded, get_sync])
def test_clean_exit(get):
    dsk = {'a': (lambda: 1 / 0,)}
    try:
        with ProgressBar() as pbar:
            get_threaded(dsk, 'a')
    except ZeroDivisionError:
        pass
    assert not pbar._running
    assert not pbar._timer.is_alive()