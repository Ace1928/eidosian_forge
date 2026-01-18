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
def object_header(num_type: int, length: int) -> bytes:
    """Return an object header for the given numeric type and text length."""
    cls = object_class(num_type)
    if cls is None:
        raise AssertionError('unsupported class type num: %d' % num_type)
    return cls.type_name + b' ' + str(length).encode('ascii') + b'\x00'