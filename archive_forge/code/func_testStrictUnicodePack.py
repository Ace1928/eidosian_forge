import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testStrictUnicodePack():
    with pytest.raises(UnicodeEncodeError):
        packb('abcídef', encoding='ascii', unicode_errors='strict')