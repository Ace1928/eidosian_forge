import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def delta_chunk_apply(dc, bbuf, write):
    """Apply own data to the target buffer
    :param bbuf: buffer providing source bytes for copy operations
    :param write: write method to call with data to write"""
    if dc.data is None:
        write(bbuf[dc.so:dc.so + dc.ts])
    elif dc.ts < len(dc.data):
        write(dc.data[:dc.ts])
    else:
        write(dc.data)