import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def matrix2str(m: Matrix) -> str:
    a, b, c, d, e, f = m
    return '[{:.2f},{:.2f},{:.2f},{:.2f}, ({:.2f},{:.2f})]'.format(a, b, c, d, e, f)