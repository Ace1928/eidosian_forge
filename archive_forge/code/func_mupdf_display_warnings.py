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
@staticmethod
def mupdf_display_warnings(on=None):
    """
        Set MuPDF warnings display to True or False.
        """
    global JM_mupdf_show_warnings
    if on is not None:
        JM_mupdf_show_warnings = bool(on)
    return JM_mupdf_show_warnings