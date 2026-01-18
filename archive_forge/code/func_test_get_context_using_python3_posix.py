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
@pytest.mark.skipif(sys.platform == 'win32', reason="Windows doesn't support different contexts")
def test_get_context_using_python3_posix():
    """get_context() respects configuration.

    If default context is changed this test will need to change too.
    """
    assert get_context() is multiprocessing.get_context('spawn')
    with dask.config.set({'multiprocessing.context': 'forkserver'}):
        assert get_context() is multiprocessing.get_context('forkserver')
    with dask.config.set({'multiprocessing.context': 'fork'}):
        assert get_context() is multiprocessing.get_context('fork')