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
def test_custom_context_used_python3_posix():
    """The 'multiprocessing.context' config is used to create the pool.

    We assume default is 'spawn', and therefore test for 'fork'.
    """

    def check_for_pytest():
        import sys
        return 'FAKE_MODULE_FOR_TEST' in sys.modules
    import sys
    sys.modules['FAKE_MODULE_FOR_TEST'] = 1
    try:
        with dask.config.set({'multiprocessing.context': 'fork'}):
            result = get({'x': (check_for_pytest,)}, 'x')
        assert result
    finally:
        del sys.modules['FAKE_MODULE_FOR_TEST']