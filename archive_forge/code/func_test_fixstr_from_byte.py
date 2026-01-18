from array import array
from srsly.msgpack import packb, unpackb
def test_fixstr_from_byte():
    _runtest('B', 1, b'\xa1', b'', False)
    _runtest('B', 31, b'\xbf', b'', False)