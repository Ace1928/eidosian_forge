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
def set_subset_fontnames(on=None):
    """
        Set / unset returning fontnames with their subset prefix.
        """
    global g_subset_fontnames
    if on is not None:
        g_subset_fontnames = bool(on)
    return g_subset_fontnames