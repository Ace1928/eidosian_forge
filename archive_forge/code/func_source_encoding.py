from __future__ import annotations
import ast
import io
import keyword
import re
import sys
import token
import tokenize
from typing import Iterable
from coverage import env
from coverage.types import TLineNo, TSourceTokenLines
def source_encoding(source: bytes) -> str:
    """Determine the encoding for `source`, according to PEP 263.

    `source` is a byte string: the text of the program.

    Returns a string, the name of the encoding.

    """
    readline = iter(source.splitlines(True)).__next__
    return tokenize.detect_encoding(readline)[0]