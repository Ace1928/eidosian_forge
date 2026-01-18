from srsly.msgpack import packb, unpackb, ExtType
def test_bin16():
    header = b'\xc5'
    data = b'x' * 256
    b = packb(data, use_bin_type=True)
    assert len(b) == len(data) + 3
    assert b[0:1] == header
    assert b[1:3] == b'\x01\x00'
    assert b[3:] == data
    assert unpackb(b) == data
    data = b'x' * 65535
    b = packb(data, use_bin_type=True)
    assert len(b) == len(data) + 3
    assert b[0:1] == header
    assert b[1:3] == b'\xff\xff'
    assert b[3:] == data
    assert unpackb(b) == data