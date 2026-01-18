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
def test_cairo_version():
    major, minor, micro = map(int, cairo_version_string().split('.'))
    assert cairo_version() == major * 10000 + minor * 100 + micro