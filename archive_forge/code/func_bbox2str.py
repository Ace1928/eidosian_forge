import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def bbox2str(bbox: Rect) -> str:
    x0, y0, x1, y1 = bbox
    return '{:.3f},{:.3f},{:.3f},{:.3f}'.format(x0, y0, x1, y1)