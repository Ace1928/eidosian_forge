import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
@pytest.mark.parametrize('compression', [pytest.param('bz2', marks=pytest.mark.xfail(raises=pa.lib.ArrowNotImplementedError)), 'brotli', 'gzip', 'lz4', 'zstd', 'snappy'])
def test_compress_decompress(compression):
    if not Codec.is_available(compression):
        pytest.skip('{} support is not built'.format(compression))
    INPUT_SIZE = 10000
    test_data = np.random.randint(0, 255, size=INPUT_SIZE).astype(np.uint8).tobytes()
    test_buf = pa.py_buffer(test_data)
    compressed_buf = pa.compress(test_buf, codec=compression)
    compressed_bytes = pa.compress(test_data, codec=compression, asbytes=True)
    assert isinstance(compressed_bytes, bytes)
    decompressed_buf = pa.decompress(compressed_buf, INPUT_SIZE, codec=compression)
    decompressed_bytes = pa.decompress(compressed_bytes, INPUT_SIZE, codec=compression, asbytes=True)
    assert isinstance(decompressed_bytes, bytes)
    assert decompressed_buf.equals(test_buf)
    assert decompressed_bytes == test_data
    with pytest.raises(ValueError):
        pa.decompress(compressed_bytes, codec=compression)