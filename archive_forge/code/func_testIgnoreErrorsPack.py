import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testIgnoreErrorsPack():
    re = unpackb(packb('abcФФФdef', encoding='ascii', unicode_errors='ignore'), raw=False, use_list=1)
    assert re == 'abcdef'