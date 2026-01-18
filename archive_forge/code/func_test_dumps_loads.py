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
def test_dumps_loads():
    with dask.config.set(func_dumps=pickle.dumps, func_loads=pickle.loads):
        assert get({'x': 1, 'y': (add, 'x', 2)}, 'y') == 3