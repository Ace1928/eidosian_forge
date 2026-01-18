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
@with_dev_shm
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_on_large_enough_dev_shm(factory):
    """Check that memmapping uses /dev/shm when possible"""
    orig_size = jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE
    try:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(32000000.0)
        p = factory(3, max_nbytes=10)
        try:
            pool_temp_folder = p._temp_folder
            folder_prefix = '/dev/shm/joblib_memmapping_folder_'
            assert pool_temp_folder.startswith(folder_prefix)
            assert os.path.exists(pool_temp_folder)
            a = np.ones(100, dtype=np.float64)
            assert a.nbytes == 800
            p.map(id, [a] * 10)
            assert len(os.listdir(pool_temp_folder)) == 1
            b = np.ones(100, dtype=np.float64) * 2
            assert b.nbytes == 800
            p.map(id, [b] * 10)
            assert len(os.listdir(pool_temp_folder)) == 2
        finally:
            p.terminate()
            del p
        for i in range(100):
            if not os.path.exists(pool_temp_folder):
                break
            sleep(0.1)
        else:
            raise AssertionError('temporary folder of pool was not deleted')
    finally:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = orig_size