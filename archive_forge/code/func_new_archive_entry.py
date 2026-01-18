from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@contextmanager
def new_archive_entry():
    entry_p = ffi.entry_new()
    try:
        yield entry_p
    finally:
        ffi.entry_free(entry_p)