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
@parametrize('data', [b'a little data as bytes.', 10000 * '{}'.format(random.randint(0, 1000) * 1000).encode('latin-1')], ids=['a little data as bytes.', 'a large data as bytes.'])
@parametrize('compress_level', [1, 3, 9])
def test_binary_zlibfile(tmpdir, data, compress_level):
    filename = tmpdir.join('test.pkl').strpath
    with open(filename, 'wb') as f:
        with BinaryZlibFile(f, 'wb', compresslevel=compress_level) as fz:
            assert fz.writable()
            fz.write(data)
            assert fz.fileno() == f.fileno()
            with raises(io.UnsupportedOperation):
                fz._check_can_read()
            with raises(io.UnsupportedOperation):
                fz._check_can_seek()
        assert fz.closed
        with raises(ValueError):
            fz._check_not_closed()
    with open(filename, 'rb') as f:
        with BinaryZlibFile(f) as fz:
            assert fz.readable()
            assert fz.seekable()
            assert fz.fileno() == f.fileno()
            assert fz.read() == data
            with raises(io.UnsupportedOperation):
                fz._check_can_write()
            assert fz.seekable()
            fz.seek(0)
            assert fz.tell() == 0
        assert fz.closed
    with BinaryZlibFile(filename, 'wb', compresslevel=compress_level) as fz:
        assert fz.writable()
        fz.write(data)
    with BinaryZlibFile(filename, 'rb') as fz:
        assert fz.read() == data
        assert fz.seekable()
    fz = BinaryZlibFile(filename, 'wb', compresslevel=compress_level)
    assert fz.writable()
    fz.write(data)
    fz.close()
    fz = BinaryZlibFile(filename, 'rb')
    assert fz.read() == data
    fz.close()