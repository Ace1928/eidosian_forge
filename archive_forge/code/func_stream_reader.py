from contextlib import contextmanager
from ctypes import cast, c_void_p, POINTER, create_string_buffer
from os import fstat, stat
from . import ffi
from .ffi import (
from .entry import ArchiveEntry, PassedArchiveEntry
@contextmanager
def stream_reader(stream, format_name='all', filter_name='all', block_size=page_size, passphrase=None, header_codec='utf-8'):
    """Read an archive from a stream.

    The `stream` object must support the standard `readinto` method.

    If `stream.seekable()` returns `True`, then an appropriate seek callback is
    passed to libarchive.
    """
    buf = create_string_buffer(block_size)
    buf_p = cast(buf, c_void_p)

    def read_func(archive_p, context, ptrptr):
        length = stream.readinto(buf)
        ptrptr = cast(ptrptr, POINTER(c_void_p))
        ptrptr[0] = buf_p
        return length

    def seek_func(archive_p, context, offset, whence):
        stream.seek(offset, whence)
        return stream.tell()
    open_cb = NO_OPEN_CB
    read_cb = READ_CALLBACK(read_func)
    close_cb = NO_CLOSE_CB
    seek_cb = SEEK_CALLBACK(seek_func)
    with new_archive_read(format_name, filter_name, passphrase) as archive_p:
        if stream.seekable():
            ffi.read_set_seek_callback(archive_p, seek_cb)
        ffi.read_open(archive_p, None, open_cb, read_cb, close_cb)
        yield ArchiveRead(archive_p, header_codec)