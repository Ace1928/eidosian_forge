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
def test_context_as_context_manager():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 1, 1)
    context = Context(surface)
    assert context.get_source().get_rgba() == (0, 0, 0, 1)
    with context:
        context.set_source_rgb(1, 0.25, 0.5)
        assert context.get_source().get_rgba() == (1, 0.25, 0.5, 1)
    assert context.get_source().get_rgba() == (0, 0, 0, 1)
    try:
        with context:
            context.set_source_rgba(1, 0.25, 0.75, 0.5)
            assert context.get_source().get_rgba() == (1, 0.25, 0.75, 0.5)
            raise ValueError
    except ValueError:
        pass
    assert context.get_source().get_rgba() == (0, 0, 0, 1)