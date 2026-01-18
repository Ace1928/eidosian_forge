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
def pixel(argb):
    """Convert a 4-byte ARGB string to native-endian."""
    return argb