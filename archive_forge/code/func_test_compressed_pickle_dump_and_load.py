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
def test_compressed_pickle_dump_and_load(tmpdir):
    expected_list = [np.arange(5, dtype=np.dtype('<i8')), np.arange(5, dtype=np.dtype('>i8')), np.arange(5, dtype=np.dtype('<f8')), np.arange(5, dtype=np.dtype('>f8')), np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'), np.arange(256, dtype=np.uint8).tobytes(), u"C'est l'été !"]
    fname = tmpdir.join('temp.pkl.gz').strpath
    dumped_filenames = numpy_pickle.dump(expected_list, fname, compress=1)
    assert len(dumped_filenames) == 1
    result_list = numpy_pickle.load(fname)
    for result, expected in zip(result_list, expected_list):
        if isinstance(expected, np.ndarray):
            expected = _ensure_native_byte_order(expected)
            assert result.dtype == expected.dtype
            np.testing.assert_equal(result, expected)
        else:
            assert result == expected