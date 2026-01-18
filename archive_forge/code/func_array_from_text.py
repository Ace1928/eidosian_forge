from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
def array_from_text(self, msg: str) -> FSArray:
    """Returns a FSArray of the size of the window containing msg"""
    rows, columns = (self.t.height, self.t.width)
    return self.array_from_text_rc(msg, rows, columns)