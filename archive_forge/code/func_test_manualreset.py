import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def test_manualreset(sizes=[0, 5, 50, 1000]):
    packer = Packer(autoreset=False)
    for size in sizes:
        packer.pack_array_header(size)
        for i in range(size):
            packer.pack(i)
    bio = BytesIO(packer.bytes())
    unpacker = Unpacker(bio, use_list=1)
    for size in sizes:
        assert unpacker.unpack() == list(range(size))
    packer.reset()
    assert packer.bytes() == b''