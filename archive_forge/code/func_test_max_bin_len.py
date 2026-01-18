import pytest
from srsly.msgpack import packb, unpackb, Packer, Unpacker, ExtType
from srsly.msgpack import PackOverflowError, PackValueError, UnpackValueError
def test_max_bin_len():
    d = b'x' * 3
    packed = packb(d, use_bin_type=True)
    unpacker = Unpacker(max_bin_len=3)
    unpacker.feed(packed)
    assert unpacker.unpack() == d
    unpacker = Unpacker(max_bin_len=2)
    with pytest.raises(UnpackValueError):
        unpacker.feed(packed)
        unpacker.unpack()