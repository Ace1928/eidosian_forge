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
def test_memmapping_pool_for_large_arrays_in_return(factory, tmpdir):
    """Check that large arrays are not copied in memory in return"""
    assert_array_equal = np.testing.assert_array_equal
    p = factory(3, max_nbytes=10, temp_folder=tmpdir.strpath)
    try:
        res = p.apply_async(np.ones, args=(1000,))
        large = res.get()
        assert not has_shareable_memory(large)
        assert_array_equal(large, np.ones(1000))
    finally:
        p.terminate()
        del p