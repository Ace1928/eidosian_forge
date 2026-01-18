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
def test_context_mask():
    mask_surface = ImageSurface(cairocffi.FORMAT_ARGB32, 2, 2)
    context = Context(mask_surface)
    context.set_source_rgba(1, 0, 0.5, 1)
    context.rectangle(0, 0, 1, 1)
    context.fill()
    context.set_source_rgba(1, 0.5, 1, 0.5)
    context.rectangle(1, 1, 1, 1)
    context.fill()
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 4, 4)
    context = Context(surface)
    context.mask(SurfacePattern(mask_surface))
    o = pixel(b'\x00\x00\x00\x00')
    b = pixel(b'\x80\x00\x00\x00')
    B = pixel(b'\xff\x00\x00\x00')
    assert surface.get_data()[:] == B + o + o + o + o + b + o + o + o + o + o + o + o + o + o + o
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 4, 4)
    context = Context(surface)
    context.mask_surface(mask_surface, surface_x=1, surface_y=2)
    o = pixel(b'\x00\x00\x00\x00')
    b = pixel(b'\x80\x00\x00\x00')
    B = pixel(b'\xff\x00\x00\x00')
    assert surface.get_data()[:] == o + o + o + o + o + o + o + o + o + B + o + o + o + o + b + o