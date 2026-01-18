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
def test_joblib_pickle_across_python_versions():
    expected_list = [np.arange(5, dtype=np.dtype('<i8')), np.arange(5, dtype=np.dtype('<f8')), np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'), np.arange(256, dtype=np.uint8).tobytes(), np.matrix([0, 1, 2], dtype=np.dtype('<i8')), u"C'est l'été !"]
    test_data_dir = os.path.dirname(os.path.abspath(data.__file__))
    pickle_extensions = ('.pkl', '.gz', '.gzip', '.bz2', 'lz4')
    if lzma is not None:
        pickle_extensions += ('.xz', '.lzma')
    pickle_filenames = [os.path.join(test_data_dir, fn) for fn in os.listdir(test_data_dir) if any((fn.endswith(ext) for ext in pickle_extensions))]
    for fname in pickle_filenames:
        _check_pickle(fname, expected_list)