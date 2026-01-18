from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def estimate_decompression_context_size():
    """Estimate the memory size requirements for a decompressor instance.

    :return:
       Integer number of bytes.
    """
    return lib.ZSTD_estimateDCtxSize()