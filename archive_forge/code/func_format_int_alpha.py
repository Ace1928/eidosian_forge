import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def format_int_alpha(value: int) -> str:
    """Format a number as lowercase letters a-z, aa-zz, etc."""
    assert value > 0
    result: List[str] = []
    while value != 0:
        value, remainder = divmod(value - 1, len(string.ascii_lowercase))
        result.append(string.ascii_lowercase[remainder])
    result.reverse()
    return ''.join(result)