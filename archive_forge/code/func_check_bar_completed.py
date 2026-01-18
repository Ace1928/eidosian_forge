from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def check_bar_completed(capsys, width=40):
    out, err = capsys.readouterr()
    assert out.count('100% Completed') == 1
    bar, percent, time = (i.strip() for i in out.split('\r')[-1].split('|'))
    assert bar == '[' + '#' * width + ']'
    assert percent == '100% Completed'