from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def frame_content_size(data):
    """Obtain the decompressed size of a frame.

    The returned value is usually accurate. But strictly speaking it should
    not be trusted.

    :return:
       ``-1`` if size unknown and a non-negative integer otherwise.
    """
    data_buffer = ffi.from_buffer(data)
    size = lib.ZSTD_getFrameContentSize(data_buffer, len(data_buffer))
    if size == lib.ZSTD_CONTENTSIZE_ERROR:
        raise ZstdError('error when determining content size')
    elif size == lib.ZSTD_CONTENTSIZE_UNKNOWN:
        return -1
    else:
        return size