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
def as_legacy_object(self, compression_level: int=-1) -> bytes:
    """Return string representing the object in the experimental format."""
    return b''.join(self.as_legacy_object_chunks(compression_level=compression_level))