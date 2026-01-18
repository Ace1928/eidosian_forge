import pytest
from srsly.msgpack import packb, unpackb, Packer, Unpacker, ExtType
from srsly.msgpack import PackOverflowError, PackValueError, UnpackValueError
def test_array_header():
    packer = Packer()
    packer.pack_array_header(2 ** 32 - 1)
    with pytest.raises(PackValueError):
        packer.pack_array_header(2 ** 32)