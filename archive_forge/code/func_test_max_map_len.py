import pytest
from srsly.msgpack import packb, unpackb, Packer, Unpacker, ExtType
from srsly.msgpack import PackOverflowError, PackValueError, UnpackValueError
def test_max_map_len():
    d = {1: 2, 3: 4, 5: 6}
    packed = packb(d)
    unpacker = Unpacker(max_map_len=3)
    unpacker.feed(packed)
    assert unpacker.unpack() == d
    unpacker = Unpacker(max_map_len=2)
    with pytest.raises(UnpackValueError):
        unpacker.feed(packed)
        unpacker.unpack()