import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def test_get_buffer():
    packer = Packer(autoreset=0, use_bin_type=True)
    packer.pack([1, 2])
    strm = BytesIO()
    strm.write(packer.getbuffer())
    written = strm.getvalue()
    expected = packb([1, 2], use_bin_type=True)
    assert written == expected