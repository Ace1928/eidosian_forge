import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
def initAndGrow(self, out, max_length):
    self.max_length = max_length
    if 0 <= max_length < self.BUFFER_BLOCK_SIZE[0]:
        block_size = max_length
    else:
        block_size = self.BUFFER_BLOCK_SIZE[0]
    block = _new_nonzero('char[]', block_size)
    if block == ffi.NULL:
        raise MemoryError
    self.list = [block]
    self.allocated = block_size
    out.dst = block
    out.size = block_size
    out.pos = 0