from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def frame_progression(self):
    """
        Return information on how much work the compressor has done.

        Returns a 3-tuple of (ingested, consumed, produced).

        >>> cctx = zstandard.ZstdCompressor()
        >>> (ingested, consumed, produced) = cctx.frame_progression()
        """
    progression = lib.ZSTD_getFrameProgression(self._cctx)
    return (progression.ingested, progression.consumed, progression.produced)