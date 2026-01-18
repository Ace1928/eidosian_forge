from srsly.msgpack import packb, unpackb, ExtType
def test_bin8():
    header = b'\xc4'
    data = b''
    b = packb(data, use_bin_type=True)
    assert len(b) == len(data) + 2
    assert b[0:2] == header + b'\x00'
    assert b[2:] == data
    assert unpackb(b) == data
    data = b'x' * 255
    b = packb(data, use_bin_type=True)
    assert len(b) == len(data) + 2
    assert b[0:2] == header + b'\xff'
    assert b[2:] == data
    assert unpackb(b) == data