from contextlib import contextmanager
from ctypes import byref, c_longlong, c_size_t, c_void_p
import os
from .ffi import (
from .read import fd_reader, file_reader, memory_reader
def extract_fd(fd, flags=None):
    """Extracts an archive from a file descriptor into the current directory.
    """
    with fd_reader(fd) as archive:
        extract_entries(archive, flags)