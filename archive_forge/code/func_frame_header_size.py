from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def frame_header_size(data):
    """Obtain the size of a frame header.

    :return:
       Integer size in bytes.
    """
    data_buffer = ffi.from_buffer(data)
    zresult = lib.ZSTD_frameHeaderSize(data_buffer, len(data_buffer))
    if lib.ZSTD_isError(zresult):
        raise ZstdError('could not determine frame header size: %s' % _zstd_error(zresult))
    return zresult