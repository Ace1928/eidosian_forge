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
def test_file_handle_persistence(tmpdir):
    objs = [np.random.random((10, 10)), 'some data']
    fobjs = [bz2.BZ2File, gzip.GzipFile]
    if lzma is not None:
        fobjs += [lzma.LZMAFile]
    filename = tmpdir.join('test.pkl').strpath
    for obj in objs:
        for fobj in fobjs:
            with fobj(filename, 'wb') as f:
                numpy_pickle.dump(obj, f)
            with fobj(filename, 'rb') as f:
                obj_reloaded = numpy_pickle.load(f)
            with open(filename, 'rb') as f:
                obj_reloaded_2 = numpy_pickle.load(f)
            if isinstance(obj, np.ndarray):
                np.testing.assert_array_equal(obj_reloaded, obj)
                np.testing.assert_array_equal(obj_reloaded_2, obj)
            else:
                assert obj_reloaded == obj
                assert obj_reloaded_2 == obj