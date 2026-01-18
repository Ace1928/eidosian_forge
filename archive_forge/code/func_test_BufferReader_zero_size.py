import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_BufferReader_zero_size():
    arr, cbuf = make_random_buffer(size=0, target='device')
    reader = cuda.BufferReader(cbuf)
    reader.seek(0)
    data = reader.read()
    assert len(data) == 0
    assert reader.tell() == 0
    buf2 = reader.read_buffer()
    arr2 = np.frombuffer(buf2.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)