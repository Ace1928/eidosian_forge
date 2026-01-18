from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def memory_size(self):
    """Size of decompression context, in bytes.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> size = dctx.memory_size()
        """
    return lib.ZSTD_sizeof_DCtx(self._dctx)