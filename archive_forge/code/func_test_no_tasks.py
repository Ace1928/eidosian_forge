from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def test_no_tasks(capsys):
    with ProgressBar():
        get_threaded({'x': 1}, 'x')
    check_bar_completed(capsys)