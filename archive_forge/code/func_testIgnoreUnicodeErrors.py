import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testIgnoreUnicodeErrors():
    re = unpackb(packb(b'abc\xeddef'), encoding='utf-8', unicode_errors='ignore', use_list=1)
    assert re == 'abcdef'