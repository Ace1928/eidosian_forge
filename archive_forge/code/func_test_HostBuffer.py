import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_HostBuffer(size):
    arr, buf = make_random_buffer(size)
    assert arr.tobytes() == buf.to_pybytes()
    hbuf = cuda.new_host_buffer(size)
    np.frombuffer(hbuf, dtype=np.uint8)[:] = arr
    assert hbuf.size == size
    assert hbuf.is_cpu
    assert arr.tobytes() == hbuf.to_pybytes()
    for i in range(size):
        assert hbuf[i] == arr[i]
    for s in [slice(None), slice(size // 4, size // 2)]:
        assert hbuf[s].to_pybytes() == arr[s].tobytes()
    sbuf = hbuf.slice(size // 4, size // 2)
    assert sbuf.parent == hbuf
    del hbuf
    with pytest.raises(TypeError, match="Do not call HostBuffer's constructor directly"):
        cuda.HostBuffer()