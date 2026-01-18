from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
def fmtstr_to_stdout_xform(self) -> Callable[[FmtStr], str]:

    def for_stdout(s: FmtStr) -> str:
        return str(s)
    return for_stdout