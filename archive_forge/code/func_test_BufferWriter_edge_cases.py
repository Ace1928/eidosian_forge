import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_BufferWriter_edge_cases():
    size = 1000
    cbuf = global_context.new_buffer(size)
    writer = cuda.BufferWriter(cbuf)
    arr, buf = make_random_buffer(size=size, target='host')
    assert writer.buffer_size == 0
    writer.buffer_size = 100
    assert writer.buffer_size == 100
    writer.write(buf.slice(length=0))
    assert writer.tell() == 0
    writer.write(buf.slice(length=10))
    writer.buffer_size = 200
    assert writer.buffer_size == 200
    assert writer.num_bytes_buffered == 0
    writer.write(buf.slice(offset=10, length=300))
    assert writer.num_bytes_buffered == 0
    writer.write(buf.slice(offset=310, length=200))
    assert writer.num_bytes_buffered == 0
    writer.write(buf.slice(offset=510, length=390))
    writer.write(buf.slice(offset=900, length=100))
    writer.flush()
    buf2 = cbuf.copy_to_host()
    assert buf2.size == size
    arr2 = np.frombuffer(buf2, dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)