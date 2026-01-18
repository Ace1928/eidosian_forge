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
def sRGB_to_pdf(srgb: int) -> tuple:
    """Convert sRGB color code to a PDF color triple.

    There is **no error checking** for performance reasons!

    Args:
        srgb: (int) RRGGBB (red, green, blue), each color in range(255).
    Returns:
        Tuple (red, green, blue) each item in intervall 0 <= item <= 1.
    """
    t = sRGB_to_rgb(srgb)
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)