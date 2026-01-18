import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testStrictUnicodeUnpack():
    with pytest.raises(UnicodeDecodeError):
        unpackb(packb(b'abc\xeddef'), raw=False, use_list=1)