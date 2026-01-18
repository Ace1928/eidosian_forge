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
def check_hexsha(hex, error_msg):
    """Check if a string is a valid hex sha string.

    Args:
      hex: Hex string to check
      error_msg: Error message to use in exception
    Raises:
      ObjectFormatException: Raised when the string is not valid
    """
    if not valid_hexsha(hex):
        raise ObjectFormatException(f'{error_msg} {hex}')