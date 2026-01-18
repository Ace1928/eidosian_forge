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
@parametrize('extension,cmethod', [('.z', 'zlib'), ('.gz', 'gzip'), ('.bz2', 'bz2'), ('.lzma', 'lzma'), ('.xz', 'xz'), ('.pkl', 'not-compressed'), ('', 'not-compressed')])
def test_compression_using_file_extension(tmpdir, extension, cmethod):
    if cmethod in ('lzma', 'xz') and lzma is None:
        pytest.skip('lzma is missing')
    filename = tmpdir.join('test.pkl').strpath
    obj = 'object to dump'
    dump_fname = filename + extension
    numpy_pickle.dump(obj, dump_fname)
    with open(dump_fname, 'rb') as f:
        assert _detect_compressor(f) == cmethod
    obj_reloaded = numpy_pickle.load(dump_fname)
    assert isinstance(obj_reloaded, type(obj))
    assert obj_reloaded == obj