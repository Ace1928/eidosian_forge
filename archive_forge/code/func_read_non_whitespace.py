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
def read_non_whitespace(stream: StreamType) -> bytes:
    """
    Find and read the next non-whitespace character (ignores whitespace).

    Args:
        stream: The data stream from which was read.

    Returns:
        The data which was read.
    """
    tok = stream.read(1)
    while tok in WHITESPACES:
        tok = stream.read(1)
    return tok