from contextlib import contextmanager
from ctypes import byref, c_longlong, c_size_t, c_void_p
import os
from .ffi import (
from .read import fd_reader, file_reader, memory_reader
def extract_file(filepath, flags=None):
    """Extracts an archive from a file into the current directory."""
    with file_reader(filepath) as archive:
        extract_entries(archive, flags)