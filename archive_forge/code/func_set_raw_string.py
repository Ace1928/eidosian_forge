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
def set_raw_string(self, text: bytes, sha: Optional[ObjectID]=None) -> None:
    """Set the contents of this object from a serialized string."""
    if not isinstance(text, bytes):
        raise TypeError('Expected bytes for text, got %r' % text)
    self.set_raw_chunks([text], sha)