from __future__ import annotations
import re
import typing as t
from dataclasses import dataclass
from enum import auto
from enum import Enum
from ..datastructures import Headers
from ..exceptions import RequestEntityTooLarge
from ..http import parse_options_header
def last_newline(self, data: bytes) -> int:
    try:
        last_nl = data.rindex(b'\n')
    except ValueError:
        last_nl = len(data)
    try:
        last_cr = data.rindex(b'\r')
    except ValueError:
        last_cr = len(data)
    return min(last_nl, last_cr)