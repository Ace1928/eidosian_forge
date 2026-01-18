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
def strip_not_given(obj: object | None) -> object:
    """Remove all top-level keys where their values are instances of `NotGiven`"""
    if obj is None:
        return None
    if not is_mapping(obj):
        return obj
    return {key: value for key, value in obj.items() if not isinstance(value, NotGiven)}