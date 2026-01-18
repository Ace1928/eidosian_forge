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
def check_time(time_seconds):
    """Check if the specified time is not prone to overflow error.

    This will raise an exception if the time is not valid.

    Args:
      time_seconds: time in seconds

    """
    if time_seconds > MAX_TIME:
        raise ObjectFormatException('Date field should not exceed %s' % MAX_TIME)