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
def test_from_null_pointer():
    for class_ in [Surface, Context, Pattern, FontFace, ScaledFont]:
        with pytest.raises(ValueError):
            class_._from_pointer(cairocffi.ffi.NULL, 'unused')