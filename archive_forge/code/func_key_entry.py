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
def key_entry(entry) -> bytes:
    """Sort key for tree entry.

    Args:
      entry: (name, value) tuple
    """
    name, value = entry
    if stat.S_ISDIR(value[0]):
        name += b'/'
    return name