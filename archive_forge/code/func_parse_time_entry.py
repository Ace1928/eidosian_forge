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
def parse_time_entry(value):
    """Parse event.

    Args:
      value: Bytes representing a git commit/tag line
    Raises:
      ObjectFormatException in case of parsing error (malformed
      field date)
    Returns: Tuple of (author, time, (timezone, timezone_neg_utc))
    """
    try:
        sep = value.rindex(b'> ')
    except ValueError:
        return (value, None, (None, False))
    try:
        person = value[0:sep + 1]
        rest = value[sep + 2:]
        timetext, timezonetext = rest.rsplit(b' ', 1)
        time = int(timetext)
        timezone, timezone_neg_utc = parse_timezone(timezonetext)
    except ValueError as exc:
        raise ObjectFormatException(exc) from exc
    return (person, time, (timezone, timezone_neg_utc))