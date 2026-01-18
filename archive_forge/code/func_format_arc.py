from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
def format_arc(value: tuple[int, int]) -> str:
    """Format an arc tuple as a string."""
    return '%d:%d' % value