import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('dest_ctx', ['same', 'another'])
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_copy_from_device(dest_ctx, size):
    arr, buf = make_random_buffer(size=size, target='device')
    lst = arr.tolist()
    if dest_ctx == 'another':
        dest_ctx = global_context1
        if buf.context.device_number == dest_ctx.device_number:
            pytest.skip('not a multi-GPU system')
    else:
        dest_ctx = buf.context
    dbuf = dest_ctx.new_buffer(size)

    def put(*args, **kwargs):
        dbuf.copy_from_device(buf, *args, **kwargs)
        rbuf = dbuf.copy_to_host()
        return np.frombuffer(rbuf, dtype=np.uint8).tolist()
    assert put() == lst
    if size > 4:
        assert put(position=size // 4) == lst[:size // 4] + lst[:-size // 4]
        assert put() == lst
        assert put(position=1, nbytes=size // 2) == lst[:1] + lst[:size // 2] + lst[-(size - size // 2 - 1):]
    for position, nbytes in [(size + 2, -1), (-2, -1), (size + 1, 0), (-3, 0)]:
        with pytest.raises(ValueError, match='position argument is out-of-range'):
            put(position=position, nbytes=nbytes)
    for position, nbytes in [(0, size + 1)]:
        with pytest.raises(ValueError, match='requested more to copy than available from device buffer'):
            put(position=position, nbytes=nbytes)
    if size < 4:
        return
    for position, nbytes in [(size // 2, (size + 1) // 2 + 1)]:
        with pytest.raises(ValueError, match='requested more to copy than available in device buffer'):
            put(position=position, nbytes=nbytes)