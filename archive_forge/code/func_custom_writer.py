from contextlib import contextmanager
from ctypes import byref, cast, c_char, c_size_t, c_void_p, POINTER
from posixpath import join
import warnings
from . import ffi
from .entry import ArchiveEntry, FileType
from .ffi import (
@contextmanager
def custom_writer(write_func, format_name, filter_name=None, open_func=None, close_func=None, block_size=page_size, archive_write_class=ArchiveWrite, options='', passphrase=None, header_codec='utf-8'):
    """Create an archive and send it in chunks to the `write_func` function.

    For formats and filters, see `WRITE_FORMATS` and `WRITE_FILTERS` in the
    `libarchive.ffi` module.
    """

    def write_cb_internal(archive_p, context, buffer_, length):
        data = cast(buffer_, POINTER(c_char * length))[0]
        return write_func(data)
    open_cb = OPEN_CALLBACK(open_func) if open_func else NO_OPEN_CB
    write_cb = WRITE_CALLBACK(write_cb_internal)
    close_cb = CLOSE_CALLBACK(close_func) if close_func else NO_CLOSE_CB
    with new_archive_write(format_name, filter_name, options, passphrase) as archive_p:
        ffi.write_set_bytes_in_last_block(archive_p, 1)
        ffi.write_set_bytes_per_block(archive_p, block_size)
        ffi.write_open(archive_p, None, open_cb, write_cb, close_cb)
        yield archive_write_class(archive_p, header_codec)