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
def test_memmapping_on_too_small_dev_shm(factory):
    orig_size = jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE
    try:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(4.2e+19)
        p = factory(3, max_nbytes=10)
        try:
            pool_temp_folder = p._temp_folder
            assert not pool_temp_folder.startswith('/dev/shm')
        finally:
            p.terminate()
            del p
        assert not os.path.exists(pool_temp_folder)
    finally:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = orig_size