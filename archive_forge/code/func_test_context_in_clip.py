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
@pytest.mark.xfail(cairo_version() < 11000, reason='Cairo version too low')
def test_context_in_clip():
    surface = ImageSurface(cairocffi.FORMAT_A8, 4, 4)
    context = Context(surface)
    context.rectangle(1, 1, 2, 2)
    assert context.in_clip(0.5, 2) is True
    assert context.in_clip(1.5, 2) is True
    context.clip()
    assert context.in_clip(0.5, 2) is False
    assert context.in_clip(1.5, 2) is True