from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
def parse_arc(value: str) -> tuple[int, int]:
    """Parse an arc string into a tuple."""
    first, last = tuple(map(int, value.split(':')))
    return (first, last)