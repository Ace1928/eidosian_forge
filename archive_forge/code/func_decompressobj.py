from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def decompressobj(self, write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE, read_across_frames=False):
    """Obtain a standard library compatible incremental decompressor.

        See :py:class:`ZstdDecompressionObj` for more documentation
        and usage examples.

        :param write_size: size of internal output buffer to collect decompressed
          chunks in.
        :param read_across_frames: whether to read across multiple zstd frames.
          If False, reading stops after 1 frame and subsequent decompress
          attempts will raise an exception.
        :return:
           :py:class:`zstandard.ZstdDecompressionObj`
        """
    if write_size < 1:
        raise ValueError('write_size must be positive')
    self._ensure_dctx()
    return ZstdDecompressionObj(self, write_size=write_size, read_across_frames=read_across_frames)