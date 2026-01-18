from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def str_to_list(s: str) -> list[str]:
    """Convert string to list."""
    if isinstance(s, str):
        return s.split(',')
    return s