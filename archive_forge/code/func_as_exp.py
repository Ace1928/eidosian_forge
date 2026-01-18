from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
def as_exp(s: str) -> str:
    """
            Float string s as in exponential format
            """
    return s if 'e' in s else '{:1.0e}'.format(float(s))