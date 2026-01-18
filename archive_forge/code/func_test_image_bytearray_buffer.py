import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_image_bytearray_buffer():
    if '__pypy__' in sys.modules:
        pytest.xfail()
    data = bytearray(800)
    surface = ImageSurface.create_for_data(data, cairocffi.FORMAT_ARGB32, 10, 20, stride=40)
    Context(surface).paint_with_alpha(0.5)
    assert data == pixel(b'\x80\x00\x00\x00') * 200