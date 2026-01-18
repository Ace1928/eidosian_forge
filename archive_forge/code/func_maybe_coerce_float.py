from __future__ import annotations
import os
import re
import inspect
import functools
from typing import (
from pathlib import Path
from typing_extensions import TypeGuard
import sniffio
from .._types import Headers, NotGiven, FileTypes, NotGivenOr, HeadersLike
from .._compat import parse_date as parse_date, parse_datetime as parse_datetime
def maybe_coerce_float(val: str | None) -> float | None:
    if val is None:
        return None
    return coerce_float(val)