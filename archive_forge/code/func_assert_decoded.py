import base64
import sys
import zlib
import pytest
from . import constants, pixbuf
def assert_decoded(surface, format_=constants.FORMAT_ARGB32, rgba=b'\x80\x00@\x80'):
    assert surface.get_width() == 3
    assert surface.get_height() == 2
    assert surface.get_format() == format_
    if sys.byteorder == 'little':
        rgba = rgba[::-1]
    assert surface.get_data()[:] == rgba * 6