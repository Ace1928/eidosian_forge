from contextlib import contextmanager
from ctypes import (
import libarchive
import libarchive.ffi as ffi
from fsspec import open_files
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.utils import DEFAULT_BLOCK_SIZE
def seek_func(archive_p, context, offset, whence):
    file.seek(offset, whence)
    return file.tell()