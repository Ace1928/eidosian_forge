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
def skip_over_whitespace(stream: StreamType) -> bool:
    """
    Similar to read_non_whitespace, but return a boolean if more than one
    whitespace character was read.

    Args:
        stream: The data stream from which was read.

    Returns:
        True if more than one whitespace was skipped, otherwise return False.
    """
    tok = WHITESPACES[0]
    cnt = 0
    while tok in WHITESPACES:
        tok = stream.read(1)
        cnt += 1
    return cnt > 1