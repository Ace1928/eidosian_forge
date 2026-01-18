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
def test_image_surface():
    assert ImageSurface.format_stride_for_width(cairocffi.FORMAT_ARGB32, 100) == 400
    assert ImageSurface.format_stride_for_width(cairocffi.FORMAT_A8, 100) == 100
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 20, 30)
    assert surface.get_format() == cairocffi.FORMAT_ARGB32
    assert surface.get_width() == 20
    assert surface.get_height() == 30
    assert surface.get_stride() == 20 * 4
    with pytest.raises(ValueError):
        data = array.array('B', b'\x00' * 799)
        ImageSurface.create_for_data(data, cairocffi.FORMAT_ARGB32, 10, 20)
    data = array.array('B', b'\x00' * 800)
    surface = ImageSurface.create_for_data(data, cairocffi.FORMAT_ARGB32, 10, 20, stride=40)
    context = Context(surface)
    assert context.get_source().get_rgba() == (0, 0, 0, 1)
    context.paint_with_alpha(0.5)
    assert data.tobytes() == pixel(b'\x80\x00\x00\x00') * 200