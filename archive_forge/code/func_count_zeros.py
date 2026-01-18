from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
def count_zeros(s):
    match = self.trailling_zeros_pattern.search(s)
    if match:
        return len(match.group(1))
    else:
        return 0