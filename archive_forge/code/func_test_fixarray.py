from srsly.msgpack import packb, unpackb
def test_fixarray():
    check_array(1, 0)
    check_array(1, (1 << 4) - 1)