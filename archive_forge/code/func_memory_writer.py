from contextlib import contextmanager
from ctypes import byref, cast, c_char, c_size_t, c_void_p, POINTER
from posixpath import join
import warnings
from . import ffi
from .entry import ArchiveEntry, FileType
from .ffi import (
@contextmanager
def memory_writer(buf, format_name, filter_name=None, archive_write_class=ArchiveWrite, options='', passphrase=None, header_codec='utf-8'):
    """Create an archive and write it into a buffer.

    For formats and filters, see `WRITE_FORMATS` and `WRITE_FILTERS` in the
    `libarchive.ffi` module.
    """
    with new_archive_write(format_name, filter_name, options, passphrase) as archive_p:
        used = byref(c_size_t())
        buf_p = cast(buf, c_void_p)
        ffi.write_open_memory(archive_p, buf_p, len(buf), used)
        yield archive_write_class(archive_p, header_codec)