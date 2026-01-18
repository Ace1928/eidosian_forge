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
def test_exceptions_rise_to_top():
    dsk = {'x': 1, 'y': (bad, 'x')}
    pytest.raises(ValueError, lambda: get(dsk, 'y'))