from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def precompute_compress(self, level=0, compression_params=None):
    """Precompute a dictionary os it can be used by multiple compressors.

        Calling this method on an instance that will be used by multiple
        :py:class:`ZstdCompressor` instances will improve performance.
        """
    if level and compression_params:
        raise ValueError('must only specify one of level or compression_params')
    if not level and (not compression_params):
        raise ValueError('must specify one of level or compression_params')
    if level:
        cparams = lib.ZSTD_getCParams(level, 0, len(self._data))
    else:
        cparams = ffi.new('ZSTD_compressionParameters')
        cparams.chainLog = compression_params.chain_log
        cparams.hashLog = compression_params.hash_log
        cparams.minMatch = compression_params.min_match
        cparams.searchLog = compression_params.search_log
        cparams.strategy = compression_params.strategy
        cparams.targetLength = compression_params.target_length
        cparams.windowLog = compression_params.window_log
    cdict = lib.ZSTD_createCDict_advanced(self._data, len(self._data), lib.ZSTD_dlm_byRef, self._dict_type, cparams, lib.ZSTD_defaultCMem)
    if cdict == ffi.NULL:
        raise ZstdError('unable to precompute dictionary')
    self._cdict = ffi.gc(cdict, lib.ZSTD_freeCDict, size=lib.ZSTD_sizeof_CDict(cdict))