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
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_pool_for_large_arrays(factory, tmpdir):
    """Check that large arrays are not copied in memory"""
    assert os.listdir(tmpdir.strpath) == []
    p = factory(3, max_nbytes=40, temp_folder=tmpdir.strpath, verbose=2)
    try:
        assert os.listdir(tmpdir.strpath) == []
        assert not os.path.exists(p._temp_folder)
        small = np.ones(5, dtype=np.float32)
        assert small.nbytes == 20
        p.map(check_array, [(small, i, 1.0) for i in range(small.shape[0])])
        assert os.listdir(tmpdir.strpath) == []
        large = np.ones(100, dtype=np.float64)
        assert large.nbytes == 800
        p.map(check_array, [(large, i, 1.0) for i in range(large.shape[0])])
        assert os.path.isdir(p._temp_folder)
        dumped_filenames = os.listdir(p._temp_folder)
        assert len(dumped_filenames) == 1
        objects = np.array(['abc'] * 100, dtype='object')
        results = p.map(has_shareable_memory, [objects])
        assert not results[0]
    finally:
        p.terminate()
        for i in range(10):
            sleep(0.1)
            if not os.path.exists(p._temp_folder):
                break
        else:
            raise AssertionError('temporary folder {} was not deleted'.format(p._temp_folder))
        del p