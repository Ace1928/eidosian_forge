import os
import zlib
from io import BytesIO
from tempfile import mkstemp
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._streams import (make_stream,
def test_all_data_read_bad_checksum(self):
    COMPRESSION_LEVEL = 6
    data = np.arange(33707000).astype(np.uint8).tobytes()
    compressed_data = zlib.compress(data, COMPRESSION_LEVEL)
    compressed_data_len = len(compressed_data)
    assert_(compressed_data_len == BLOCK_SIZE + 2)
    compressed_data = compressed_data[:-1] + bytes([compressed_data[-1] + 1 & 255])
    compressed_stream = BytesIO(compressed_data)
    stream = ZlibInputStream(compressed_stream, compressed_data_len)
    assert_(not stream.all_data_read())
    stream.seek(len(data))
    assert_raises(zlib.error, stream.all_data_read)