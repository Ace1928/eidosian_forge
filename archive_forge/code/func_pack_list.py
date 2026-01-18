from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def pack_list(from_, pack_type):
    """ Return the wire packed version of `from_`. `pack_type` should be some
    subclass of `xcffib.Struct`, or a string that can be passed to
    `struct.pack`. You must pass `size` if `pack_type` is a struct.pack string.
    """
    if len(from_) == 0:
        return bytes()
    if pack_type == 'c':
        if isinstance(from_, bytes):
            from_ = [bytes((b,)) for b in bytes(from_)]
        elif isinstance(from_, str):
            from_ = [bytes((b,)) for b in bytearray(from_, 'utf-8')]
        elif isinstance(from_[0], int):

            def to_bytes(v):
                for _ in range(4):
                    v, r = divmod(v, 256)
                    yield r
            from_ = [bytes((b,)) for i in from_ for b in to_bytes(i)]
    if isinstance(pack_type, str):
        return struct.pack('=%d%s' % (len(from_), pack_type), *from_)
    else:
        buf = io.BytesIO()
        for item in from_:
            if isinstance(item, Protobj) and hasattr(item, 'pack'):
                buf.write(item.pack())
            else:
                buf.write(item)
        return buf.getvalue()