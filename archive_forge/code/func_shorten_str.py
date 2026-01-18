import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def shorten_str(s: str, size: int) -> str:
    if size < 7:
        return s[:size]
    if len(s) > size:
        length = (size - 5) // 2
        return '{} ... {}'.format(s[:length], s[-length:])
    else:
        return s