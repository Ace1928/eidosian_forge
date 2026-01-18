from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
def as_mathtex(s: str) -> str:
    """
            Mathtex for maplotlib
            """
    if 'e' not in s:
        assert s == '1', f"Unexpected value s = {s!r}, instead of '1'"
        return f'${self.base}^{{0}}$'
    exp = s.split('e')[1]
    return f'${self.base}^{{{exp}}}$'