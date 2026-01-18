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
def jm_increase_seqno(dev, ctx, *vargs):
    try:
        dev.seqno += 1
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise