import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_copy_from_to_host(size):
    dt = np.dtype('uint16')
    nbytes = size * dt.itemsize
    buf = pa.allocate_buffer(nbytes, resizable=True)
    assert isinstance(buf, pa.Buffer)
    assert not isinstance(buf, cuda.CudaBuffer)
    arr = np.frombuffer(buf, dtype=dt)
    assert arr.size == size
    arr[:] = range(size)
    arr_ = np.frombuffer(buf, dtype=dt)
    np.testing.assert_equal(arr, arr_)
    device_buffer = global_context.new_buffer(nbytes)
    assert isinstance(device_buffer, cuda.CudaBuffer)
    assert isinstance(device_buffer, pa.Buffer)
    assert device_buffer.size == nbytes
    assert not device_buffer.is_cpu
    device_buffer.copy_from_host(buf, position=0, nbytes=nbytes)
    buf2 = device_buffer.copy_to_host(position=0, nbytes=nbytes)
    arr2 = np.frombuffer(buf2, dtype=dt)
    np.testing.assert_equal(arr, arr2)