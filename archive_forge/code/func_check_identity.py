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
def check_identity(identity: bytes, error_msg: str) -> None:
    """Check if the specified identity is valid.

    This will raise an exception if the identity is not valid.

    Args:
      identity: Identity string
      error_msg: Error message to use in exception
    """
    email_start = identity.find(b'<')
    email_end = identity.find(b'>')
    if not all([email_start >= 1, identity[email_start - 1] == b' '[0], identity.find(b'<', email_start + 1) == -1, email_end == len(identity) - 1, b'\x00' not in identity, b'\n' not in identity]):
        raise ObjectFormatException(error_msg)