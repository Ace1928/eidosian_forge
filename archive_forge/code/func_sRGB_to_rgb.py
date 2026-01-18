import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def sRGB_to_rgb(srgb: int) -> tuple:
    """Convert sRGB color code to an RGB color triple.

    There is **no error checking** for performance reasons!

    Args:
        srgb: (int) RRGGBB (red, green, blue), each color in range(255).
    Returns:
        Tuple (red, green, blue) each item in intervall 0 <= item <= 255.
    """
    r = srgb >> 16
    g = srgb - (r << 16) >> 8
    b = srgb - (r << 16) - (g << 8)
    return (r, g, b)