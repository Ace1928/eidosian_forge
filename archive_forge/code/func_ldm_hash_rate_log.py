from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
@property
def ldm_hash_rate_log(self):
    return _get_compression_parameter(self._params, lib.ZSTD_c_ldmHashRateLog)