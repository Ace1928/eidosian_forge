from array import array
from srsly.msgpack import packb, unpackb
def test_bin8_from_byte():
    _runtest('B', 1, b'\xc4', b'\x01', True)
    _runtest('B', 2 ** 8 - 1, b'\xc4', b'\xff', True)