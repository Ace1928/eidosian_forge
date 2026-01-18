from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
@classmethod
def array_from_text_rc(cls, msg: str, rows: int, columns: int) -> FSArray:
    arr = FSArray(0, columns)
    i = 0
    for c in msg:
        if i >= rows * columns:
            return arr
        elif c in '\r\n':
            i = (i // columns + 1) * columns - 1
        else:
            arr[i // arr.width, i % arr.width] = [fmtstr(c)]
        i += 1
    return arr