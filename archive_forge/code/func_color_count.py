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
def color_count(self, colors=0, clip=None):
    """
        Return count of each color.
        """
    pm = self.this
    rc = JM_color_count(pm, clip)
    if not rc:
        raise RuntimeError(MSG_COLOR_COUNT_FAILED)
    if not colors:
        return len(rc)
    return rc