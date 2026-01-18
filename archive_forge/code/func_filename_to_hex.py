import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def filename_to_hex(filename):
    """Takes an object filename and returns its corresponding hex sha."""
    names = filename.rsplit(os.path.sep, 2)[-2:]
    errmsg = 'Invalid object filename: %s' % filename
    assert len(names) == 2, errmsg
    base, rest = names
    assert len(base) == 2 and len(rest) == 38, errmsg
    hex = (base + rest).encode('ascii')
    hex_to_sha(hex)
    return hex