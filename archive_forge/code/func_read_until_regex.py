import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def read_until_regex(stream: StreamType, regex: Pattern[bytes]) -> bytes:
    """
    Read until the regular expression pattern matched (ignore the match).
    Treats EOF on the underlying stream as the end of the token to be matched.

    Args:
        regex: re.Pattern

    Returns:
        The read bytes.
    """
    name = b''
    while True:
        tok = stream.read(16)
        if not tok:
            return name
        m = regex.search(name + tok)
        if m is not None:
            stream.seek(m.start() - (len(name) + len(tok)), 1)
            name = (name + tok)[:m.start()]
            break
        name += tok
    return name