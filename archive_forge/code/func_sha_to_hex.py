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
def sha_to_hex(sha):
    """Takes a string and returns the hex of the sha within."""
    hexsha = binascii.hexlify(sha)
    assert len(hexsha) == 40, 'Incorrect length of sha1 string: %r' % hexsha
    return hexsha