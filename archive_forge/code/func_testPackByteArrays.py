import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testPackByteArrays():
    test_data = [bytearray(b''), bytearray(b'abcd'), (bytearray(b'defgh'),)]
    for td in test_data:
        check(td)