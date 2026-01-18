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
def hex_to_filename(path, hex):
    """Takes a hex sha and returns its filename relative to the given path."""
    if type(path) is not type(hex) and getattr(path, 'encode', None) is not None:
        hex = hex.decode('ascii')
    dir = hex[:2]
    file = hex[2:]
    return os.path.join(path, dir, file)