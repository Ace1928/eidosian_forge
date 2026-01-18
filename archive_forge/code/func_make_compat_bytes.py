import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def make_compat_bytes(in_str: str) -> bytes:
    """Converts to bytes, encoding to unicode."""
    assert isinstance(in_str, str), str(type(in_str))
    return in_str.encode()