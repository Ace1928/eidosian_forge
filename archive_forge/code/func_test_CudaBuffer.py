import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_CudaBuffer(size):
    arr, buf = make_random_buffer(size)
    assert arr.tobytes() == buf.to_pybytes()
    cbuf = global_context.buffer_from_data(buf)
    assert cbuf.size == size
    assert not cbuf.is_cpu
    assert arr.tobytes() == cbuf.to_pybytes()
    if size > 0:
        assert cbuf.address > 0
    for i in range(size):
        assert cbuf[i] == arr[i]
    for s in [slice(None), slice(size // 4, size // 2)]:
        assert cbuf[s].to_pybytes() == arr[s].tobytes()
    sbuf = cbuf.slice(size // 4, size // 2)
    assert sbuf.parent == cbuf
    with pytest.raises(TypeError, match="Do not call CudaBuffer's constructor directly"):
        cuda.CudaBuffer()