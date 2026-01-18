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
def check_if_whitespace_only(value: bytes) -> bool:
    """
    Check if the given value consists of whitespace characters only.

    Args:
        value: The bytes to check.

    Returns:
        True if the value only has whitespace characters, otherwise return False.
    """
    for index in range(len(value)):
        current = value[index:index + 1]
        if current not in WHITESPACES:
            return False
    return True