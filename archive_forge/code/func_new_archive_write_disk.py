from contextlib import contextmanager
from ctypes import byref, c_longlong, c_size_t, c_void_p
import os
from .ffi import (
from .read import fd_reader, file_reader, memory_reader
@contextmanager
def new_archive_write_disk(flags):
    archive_p = write_disk_new()
    write_disk_set_options(archive_p, flags)
    try:
        yield archive_p
    finally:
        write_free(archive_p)