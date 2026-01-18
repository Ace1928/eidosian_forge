from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
@staticmethod
def from_level(level, source_size=0, dict_size=0, **kwargs):
    """Create compression parameters from a compression level.

        :param level:
           Integer compression level.
        :param source_size:
           Integer size in bytes of source to be compressed.
        :param dict_size:
           Integer size in bytes of compression dictionary to use.
        :return:
           :py:class:`ZstdCompressionParameters`
        """
    params = lib.ZSTD_getCParams(level, source_size, dict_size)
    args = {'window_log': 'windowLog', 'chain_log': 'chainLog', 'hash_log': 'hashLog', 'search_log': 'searchLog', 'min_match': 'minMatch', 'target_length': 'targetLength', 'strategy': 'strategy'}
    for arg, attr in args.items():
        if arg not in kwargs:
            kwargs[arg] = getattr(params, attr)
    return ZstdCompressionParameters(**kwargs)