import copy
import os
import random
import re
import io
import sys
import warnings
import gzip
import zlib
import bz2
import pickle
import socket
from contextlib import closing
import mmap
from pathlib import Path
import pytest
from joblib.test.common import np, with_numpy, with_lz4, without_lz4
from joblib.test.common import with_memory_profiler, memory_used
from joblib.testing import parametrize, raises, warns
from joblib import numpy_pickle, register_compressor
from joblib.test import data
from joblib.numpy_pickle_utils import _IO_BUFFER_SIZE
from joblib.numpy_pickle_utils import _detect_compressor
from joblib.numpy_pickle_utils import _is_numpy_array_byte_order_mismatch
from joblib.numpy_pickle_utils import _ensure_native_byte_order
from joblib.compressor import (_COMPRESSORS, _LZ4_PREFIX, CompressorWrapper,
@with_numpy
@parametrize('protocol', protocols)
def test_memmap_alignment_padding(tmpdir, protocol):
    fname = tmpdir.join('test.mmap').strpath
    a = np.random.randn(2)
    numpy_pickle.dump(a, fname, protocol=protocol)
    memmap = numpy_pickle.load(fname, mmap_mode='r')
    assert isinstance(memmap, np.memmap)
    np.testing.assert_array_equal(a, memmap)
    assert memmap.ctypes.data % numpy_pickle.NUMPY_ARRAY_ALIGNMENT_BYTES == 0
    assert memmap.flags.aligned
    array_list = [np.random.randn(2), np.random.randn(2), np.random.randn(2), np.random.randn(2)]
    fname = tmpdir.join('test1.mmap').strpath
    numpy_pickle.dump(array_list, fname, protocol=protocol)
    l_reloaded = numpy_pickle.load(fname, mmap_mode='r')
    for idx, memmap in enumerate(l_reloaded):
        assert isinstance(memmap, np.memmap)
        np.testing.assert_array_equal(array_list[idx], memmap)
        assert memmap.ctypes.data % numpy_pickle.NUMPY_ARRAY_ALIGNMENT_BYTES == 0
        assert memmap.flags.aligned
    array_dict = {'a0': np.arange(2, dtype=np.uint8), 'a1': np.arange(3, dtype=np.uint8), 'a2': np.arange(5, dtype=np.uint8), 'a3': np.arange(7, dtype=np.uint8), 'a4': np.arange(11, dtype=np.uint8), 'a5': np.arange(13, dtype=np.uint8), 'a6': np.arange(17, dtype=np.uint8), 'a7': np.arange(19, dtype=np.uint8), 'a8': np.arange(23, dtype=np.uint8)}
    fname = tmpdir.join('test2.mmap').strpath
    numpy_pickle.dump(array_dict, fname, protocol=protocol)
    d_reloaded = numpy_pickle.load(fname, mmap_mode='r')
    for key, memmap in d_reloaded.items():
        assert isinstance(memmap, np.memmap)
        np.testing.assert_array_equal(array_dict[key], memmap)
        assert memmap.ctypes.data % numpy_pickle.NUMPY_ARRAY_ALIGNMENT_BYTES == 0
        assert memmap.flags.aligned