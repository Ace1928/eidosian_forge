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
def get_pdf_now() -> str:
    """
    "Now" timestamp in PDF Format
    """
    import time
    tz = "%s'%s'" % (str(abs(time.altzone // 3600)).rjust(2, '0'), str(abs(time.altzone // 60) % 60).rjust(2, '0'))
    tstamp = time.strftime('D:%Y%m%d%H%M%S', time.localtime())
    if time.altzone > 0:
        tstamp += '-' + tz
    elif time.altzone < 0:
        tstamp += '+' + tz
    else:
        pass
    return tstamp