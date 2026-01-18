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
@pytest.mark.xfail(cairo_version() < 11512, reason='Cairo version too low')
def test_font_options_variations():
    options = FontOptions()
    assert options.get_variations() is None
    options.set_variations('wght 400, wdth 300')
    assert options.get_variations() == 'wght 400, wdth 300'
    options.set_variations(None)
    assert options.get_variations() is None