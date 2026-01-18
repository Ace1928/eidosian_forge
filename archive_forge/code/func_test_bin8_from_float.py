from array import array
from srsly.msgpack import packb, unpackb
def test_bin8_from_float():
    _runtest('f', 4, b'\xc4', b'\x04', True)
    _runtest('f', 2 ** 8 - 4, b'\xc4', b'\xfc', True)