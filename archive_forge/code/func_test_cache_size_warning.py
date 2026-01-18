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
@parametrize('cache_size', [None, 0, 10])
def test_cache_size_warning(tmpdir, cache_size):
    filename = tmpdir.join('test.pkl').strpath
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as warninfo:
        numpy_pickle.dump(a, filename, cache_size=cache_size)
    expected_nb_warnings = 1 if cache_size is not None else 0
    assert len(warninfo) == expected_nb_warnings
    for w in warninfo:
        assert w.category == DeprecationWarning
        assert str(w.message) == "Please do not set 'cache_size' in joblib.dump, this parameter has no effect and will be removed. You used 'cache_size={0}'".format(cache_size)