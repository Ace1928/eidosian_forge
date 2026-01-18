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
def test_memmap_based_array_reducing(tmpdir):
    """Check that it is possible to reduce a memmap backed array"""
    assert_array_equal = np.testing.assert_array_equal
    filename = tmpdir.join('test.mmap').strpath
    buffer = np.memmap(filename, dtype=np.float64, shape=500, mode='w+')
    buffer[:] = -1.0 * np.arange(buffer.shape[0], dtype=buffer.dtype)
    buffer.flush()
    a = np.memmap(filename, dtype=np.float64, shape=(3, 5, 4), mode='r+', order='F', offset=4)
    a[:] = np.arange(60).reshape(a.shape)
    b = a[1:-1, 2:-1, 2:4]
    c = np.asarray(b)
    d = c.T
    reducer = ArrayMemmapForwardReducer(None, tmpdir.strpath, 'c', True)

    def reconstruct_array_or_memmap(x):
        cons, args = reducer(x)
        return cons(*args)
    a_reconstructed = reconstruct_array_or_memmap(a)
    assert has_shareable_memory(a_reconstructed)
    assert isinstance(a_reconstructed, np.memmap)
    assert_array_equal(a_reconstructed, a)
    b_reconstructed = reconstruct_array_or_memmap(b)
    assert has_shareable_memory(b_reconstructed)
    assert_array_equal(b_reconstructed, b)
    c_reconstructed = reconstruct_array_or_memmap(c)
    assert not isinstance(c_reconstructed, np.memmap)
    assert has_shareable_memory(c_reconstructed)
    assert_array_equal(c_reconstructed, c)
    d_reconstructed = reconstruct_array_or_memmap(d)
    assert not isinstance(d_reconstructed, np.memmap)
    assert has_shareable_memory(d_reconstructed)
    assert_array_equal(d_reconstructed, d)
    a3 = a * 3
    assert not has_shareable_memory(a3)
    a3_reconstructed = reconstruct_array_or_memmap(a3)
    assert not has_shareable_memory(a3_reconstructed)
    assert not isinstance(a3_reconstructed, np.memmap)
    assert_array_equal(a3_reconstructed, a * 3)
    b3 = np.asarray(a3)
    assert not has_shareable_memory(b3)
    b3_reconstructed = reconstruct_array_or_memmap(b3)
    assert isinstance(b3_reconstructed, np.ndarray)
    assert not has_shareable_memory(b3_reconstructed)
    assert_array_equal(b3_reconstructed, b3)