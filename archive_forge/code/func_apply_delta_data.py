import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def apply_delta_data(src_buf, src_buf_size, delta_buf, delta_buf_size, write):
    """
    Apply data from a delta buffer using a source buffer to the target file

    :param src_buf: random access data from which the delta was created
    :param src_buf_size: size of the source buffer in bytes
    :param delta_buf_size: size for the delta buffer in bytes
    :param delta_buf: random access delta data
    :param write: write method taking a chunk of bytes

    **Note:** transcribed to python from the similar routine in patch-delta.c"""
    i = 0
    db = delta_buf
    while i < delta_buf_size:
        c = db[i]
        i += 1
        if c & 128:
            cp_off, cp_size = (0, 0)
            if c & 1:
                cp_off = db[i]
                i += 1
            if c & 2:
                cp_off |= db[i] << 8
                i += 1
            if c & 4:
                cp_off |= db[i] << 16
                i += 1
            if c & 8:
                cp_off |= db[i] << 24
                i += 1
            if c & 16:
                cp_size = db[i]
                i += 1
            if c & 32:
                cp_size |= db[i] << 8
                i += 1
            if c & 64:
                cp_size |= db[i] << 16
                i += 1
            if not cp_size:
                cp_size = 65536
            rbound = cp_off + cp_size
            if rbound < cp_size or rbound > src_buf_size:
                break
            write(src_buf[cp_off:cp_off + cp_size])
        elif c:
            write(db[i:i + c])
            i += c
        else:
            raise ValueError('unexpected delta opcode 0')
    assert i == delta_buf_size, 'delta replay has gone wild'