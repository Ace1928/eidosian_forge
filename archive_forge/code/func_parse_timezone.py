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
def parse_timezone(text):
    """Parse a timezone text fragment (e.g. '+0100').

    Args:
      text: Text to parse.
    Returns: Tuple with timezone as seconds difference to UTC
        and a boolean indicating whether this was a UTC timezone
        prefixed with a negative sign (-0000).
    """
    if text[0] not in b'+-':
        raise ValueError('Timezone must start with + or - ({text})'.format(**vars()))
    sign = text[:1]
    offset = int(text[1:])
    if sign == b'-':
        offset = -offset
    unnecessary_negative_timezone = offset >= 0 and sign == b'-'
    signum = offset < 0 and -1 or 1
    offset = abs(offset)
    hours = int(offset / 100)
    minutes = offset % 100
    return (signum * (hours * 3600 + minutes * 60), unnecessary_negative_timezone)