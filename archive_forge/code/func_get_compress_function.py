from __future__ import absolute_import
from .snappy import (
def get_compress_function(specified_format):
    if specified_format == FORMAT_AUTO:
        return _COMPRESS_METHODS[_DEFAULT_COMPRESS_FORMAT]
    return _COMPRESS_METHODS[specified_format]