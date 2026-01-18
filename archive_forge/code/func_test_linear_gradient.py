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
def test_linear_gradient():
    gradient = LinearGradient(1, 2, 10, 20)
    assert gradient.get_linear_points() == (1, 2, 10, 20)
    gradient.add_color_stop_rgb(1, 1, 0.5, 0.25)
    gradient.add_color_stop_rgb(offset=0.5, red=1, green=0.5, blue=0.25)
    gradient.add_color_stop_rgba(0.5, 1, 0.5, 0.75, 0.25)
    assert gradient.get_color_stops() == [(0.5, 1, 0.5, 0.25, 1), (0.5, 1, 0.5, 0.75, 0.25), (1, 1, 0.5, 0.25, 1)]
    surface = ImageSurface(cairocffi.FORMAT_A8, 8, 4)
    assert surface.get_data()[:] == b'\x00' * 32
    gradient = LinearGradient(1.5, 0, 6.5, 0)
    gradient.add_color_stop_rgba(0, 0, 0, 0, 0)
    gradient.add_color_stop_rgba(1, 0, 0, 0, 1)
    context = Context(surface)
    context.set_source(gradient)
    context.paint()
    assert surface.get_data()[:] == b'\x00\x003f\x99\xcc\xff\xff' * 4
    assert b'/ShadingType 2' not in pdf_with_pattern()
    assert b'/ShadingType 2' in pdf_with_pattern(gradient)