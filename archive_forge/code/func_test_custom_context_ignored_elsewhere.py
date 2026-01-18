from __future__ import annotations
import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from operator import add
import pytest
import dask
from dask import compute, delayed
from dask.multiprocessing import _dumps, _loads, get, get_context, remote_exception
from dask.system import CPU_COUNT
from dask.utils_test import inc
@pytest.mark.skipif(sys.platform != 'win32', reason='POSIX supports different contexts')
def test_custom_context_ignored_elsewhere():
    """On Windows, setting 'multiprocessing.context' doesn't explode.

    Presumption is it's not used since it's unsupported, but mostly we care about
    not breaking anything.
    """
    assert get({'x': (inc, 1)}, 'x') == 2
    with pytest.warns(UserWarning):
        with dask.config.set({'multiprocessing.context': 'forkserver'}):
            assert get({'x': (inc, 1)}, 'x') == 2