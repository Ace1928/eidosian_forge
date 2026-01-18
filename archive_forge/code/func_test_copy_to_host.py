import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_copy_to_host(size):
    arr, dbuf = make_random_buffer(size, target='device')
    buf = dbuf.copy_to_host()
    assert buf.is_cpu
    np.testing.assert_equal(arr, np.frombuffer(buf, dtype=np.uint8))
    buf = dbuf.copy_to_host(position=size // 4)
    assert buf.is_cpu
    np.testing.assert_equal(arr[size // 4:], np.frombuffer(buf, dtype=np.uint8))
    buf = dbuf.copy_to_host(position=size // 4, nbytes=size // 8)
    assert buf.is_cpu
    np.testing.assert_equal(arr[size // 4:size // 4 + size // 8], np.frombuffer(buf, dtype=np.uint8))
    buf = dbuf.copy_to_host(position=size // 4, nbytes=0)
    assert buf.is_cpu
    assert buf.size == 0
    for position, nbytes in [(size + 2, -1), (-2, -1), (size + 1, 0), (-3, 0)]:
        with pytest.raises(ValueError, match='position argument is out-of-range'):
            dbuf.copy_to_host(position=position, nbytes=nbytes)
    for position, nbytes in [(0, size + 1), (size // 2, (size + 1) // 2 + 1), (size, 1)]:
        with pytest.raises(ValueError, match='requested more to copy than available from device buffer'):
            dbuf.copy_to_host(position=position, nbytes=nbytes)
    buf = pa.allocate_buffer(size // 4)
    dbuf.copy_to_host(buf=buf)
    np.testing.assert_equal(arr[:size // 4], np.frombuffer(buf, dtype=np.uint8))
    if size < 12:
        return
    dbuf.copy_to_host(buf=buf, position=12)
    np.testing.assert_equal(arr[12:12 + size // 4], np.frombuffer(buf, dtype=np.uint8))
    dbuf.copy_to_host(buf=buf, nbytes=12)
    np.testing.assert_equal(arr[:12], np.frombuffer(buf, dtype=np.uint8)[:12])
    dbuf.copy_to_host(buf=buf, nbytes=12, position=6)
    np.testing.assert_equal(arr[6:6 + 12], np.frombuffer(buf, dtype=np.uint8)[:12])
    for position, nbytes in [(0, size + 10), (10, size - 5), (0, size // 2), (size // 4, size // 4 + 1)]:
        with pytest.raises(ValueError, match='requested copy does not fit into host buffer'):
            dbuf.copy_to_host(buf=buf, position=position, nbytes=nbytes)