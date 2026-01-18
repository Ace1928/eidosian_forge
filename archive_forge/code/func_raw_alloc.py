import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
@ffi.def_extern()
def raw_alloc(size: int) -> object:
    if size == 0:
        return ffi.NULL
    block = ffi.new('char[]', size)
    _allocated.append(block)
    return block