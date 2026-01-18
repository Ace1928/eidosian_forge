from __future__ import annotations
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from time import sleep, time
import pytest
import dask
from dask.system import CPU_COUNT
from dask.threaded import get
from dask.utils_test import add, inc
def test_broken_callback():
    from dask.callbacks import Callback

    def _f_ok(*args, **kwargs):
        pass

    def _f_broken(*args, **kwargs):
        raise ValueError('my_exception')
    dsk = {'x': 1}
    with Callback(start=_f_broken, finish=_f_ok):
        with Callback(start=_f_ok, finish=_f_ok):
            with pytest.raises(ValueError, match='my_exception'):
                get(dsk, 'x')