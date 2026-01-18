from srsly.msgpack import packb, unpackb
def test_array32():
    check_array(5, 1 << 16)