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
def test_context_current_transform_matrix():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 1, 1)
    context = Context(surface)
    assert isinstance(context.get_matrix(), Matrix)
    assert context.get_matrix().as_tuple() == (1, 0, 0, 1, 0, 0)
    context.translate(6, 5)
    assert context.get_matrix().as_tuple() == (1, 0, 0, 1, 6, 5)
    context.scale(1, 6)
    assert context.get_matrix().as_tuple() == (1, 0, 0, 6, 6, 5)
    context.scale(0.5)
    assert context.get_matrix().as_tuple() == (0.5, 0, 0, 3, 6, 5)
    context.rotate(math.pi / 2)
    assert round_tuple(context.get_matrix().as_tuple()) == (0, 3, -0.5, 0, 6, 5)
    context.identity_matrix()
    assert context.get_matrix().as_tuple() == (1, 0, 0, 1, 0, 0)
    context.set_matrix(Matrix(2, 1, 3, 7, 8, 2))
    assert context.get_matrix().as_tuple() == (2, 1, 3, 7, 8, 2)
    context.transform(Matrix(2, 0, 0, 0.5, 0, 0))
    assert context.get_matrix().as_tuple() == (4, 2, 1.5, 3.5, 8, 2)
    context.set_matrix(Matrix(2, 0, 0, 3, 12, 4))
    assert context.user_to_device_distance(1, 2) == (2, 6)
    assert context.user_to_device(1, 2) == (14, 10)
    assert context.device_to_user_distance(2, 6) == (1, 2)
    assert round_tuple(context.device_to_user(14, 10)) == (1, 2)