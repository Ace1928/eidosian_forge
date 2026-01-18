from __future__ import annotations
import sys
from io import StringIO
from json import JSONEncoder, loads
from typing import TYPE_CHECKING
def max_width_col(table: list[list], col_idx: int) -> int:
    """Get the maximum width of the given column index."""
    return max((len(row[col_idx]) for row in table))