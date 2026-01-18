import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testPackFloat():
    assert packb(1.0, use_single_float=True) == b'\xca' + struct.pack(str('>f'), 1.0)
    assert packb(1.0, use_single_float=False) == b'\xcb' + struct.pack(str('>d'), 1.0)