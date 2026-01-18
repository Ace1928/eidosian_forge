import sys
import types
import collections
import io
from opcode import *
from opcode import (
def findlinestarts(code):
    """Find the offsets in a byte code which are start of lines in the source.

    Generate pairs (offset, lineno)
    """
    lastline = None
    for start, end, line in code.co_lines():
        if line is not None and line != lastline:
            lastline = line
            yield (start, line)
    return