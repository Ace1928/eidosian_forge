import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def get_bound(pts: Iterable[Point]) -> Rect:
    """Compute a minimal rectangle that covers all the points."""
    limit: Rect = (INF, INF, -INF, -INF)
    x0, y0, x1, y1 = limit
    for x, y in pts:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return (x0, y0, x1, y1)