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
@parametrize('bad_value', [-1, 10, 15, 'a', (), {}])
def test_binary_zlibfile_bad_compression_levels(tmpdir, bad_value):
    filename = tmpdir.join('test.pkl').strpath
    with raises(ValueError) as excinfo:
        BinaryZlibFile(filename, 'wb', compresslevel=bad_value)
    pattern = re.escape("'compresslevel' must be an integer between 1 and 9. You provided 'compresslevel={}'".format(bad_value))
    excinfo.match(pattern)