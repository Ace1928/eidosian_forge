import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def is_loose_object(m):
    """
    :return: True the file contained in memory map m appears to be a loose object.
        Only the first two bytes are needed"""
    b0, b1 = map(ord, m[:2])
    word = (b0 << 8) + b1
    return b0 == 120 and word % 31 == 0