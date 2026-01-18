from io import BytesIO
import pytest
import logging
from matplotlib import _afm
from matplotlib import font_manager as fm
def test_nonascii_str():
    inp_str = 'привет'
    byte_str = inp_str.encode('utf8')
    ret = _afm._to_str(byte_str)
    assert ret == inp_str