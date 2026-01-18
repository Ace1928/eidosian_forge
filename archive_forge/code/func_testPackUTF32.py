import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testPackUTF32():
    try:
        test_data = ['', 'abcd', ['defgh'], 'Русский текст']
        for td in test_data:
            re = unpackb(packb(td, encoding='utf-32'), use_list=1, encoding='utf-32')
            assert re == td
    except LookupError as e:
        pytest.xfail(e)