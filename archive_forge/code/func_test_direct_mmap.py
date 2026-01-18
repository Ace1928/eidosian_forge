import os
import mmap
import sys
import platform
import gc
import pickle
import itertools
from time import sleep
import subprocess
import threading
import faulthandler
import pytest
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises, parametrize, skipif
from joblib.backports import make_memmap
from joblib.parallel import Parallel, delayed
from joblib.pool import MemmappingPool
from joblib.executor import _TestingMemmappingExecutor as TestExecutor
from joblib._memmapping_reducer import has_shareable_memory
from joblib._memmapping_reducer import ArrayMemmapForwardReducer
from joblib._memmapping_reducer import _strided_from_memmap
from joblib._memmapping_reducer import _get_temp_dir
from joblib._memmapping_reducer import _WeakArrayKeyMap
from joblib._memmapping_reducer import _get_backing_memmap
import joblib._memmapping_reducer as jmr
@with_numpy
@with_multiprocessing
def test_direct_mmap(tmpdir):
    testfile = str(tmpdir.join('arr.dat'))
    a = np.arange(10, dtype='uint8')
    a.tofile(testfile)

    def _read_array():
        with open(testfile) as fd:
            mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ, offset=0)
        return np.ndarray((10,), dtype=np.uint8, buffer=mm, offset=0)

    def func(x):
        return x ** 2
    arr = _read_array()
    ref = Parallel(n_jobs=2)((delayed(func)(x) for x in [a]))
    results = Parallel(n_jobs=2)((delayed(func)(x) for x in [arr]))
    np.testing.assert_array_equal(results, ref)

    def worker():
        return _read_array()
    results = Parallel(n_jobs=2)((delayed(worker)() for _ in range(1)))
    np.testing.assert_array_equal(results[0], arr)