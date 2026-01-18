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
def raw_length(self) -> int:
    """Returns the length of the raw string of this object."""
    return sum(map(len, self.as_raw_chunks()))