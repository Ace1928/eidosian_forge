import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def is_zipfile(filename):
    """Quickly see if a file is a ZIP file by checking the magic number.

    The filename argument may be a file or file-like object too.
    """
    result = False
    try:
        if hasattr(filename, 'read'):
            result = _check_zipfile(fp=filename)
        else:
            with open(filename, 'rb') as fp:
                result = _check_zipfile(fp)
    except OSError:
        pass
    return result