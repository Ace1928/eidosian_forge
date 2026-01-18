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
def b_(s: Union[str, bytes]) -> bytes:
    if isinstance(s, bytes):
        return s
    bc = B_CACHE
    if s in bc:
        return bc[s]
    try:
        r = s.encode('latin-1')
        if len(s) < 2:
            bc[s] = r
        return r
    except Exception:
        r = s.encode('utf-8')
        if len(s) < 2:
            bc[s] = r
        return r