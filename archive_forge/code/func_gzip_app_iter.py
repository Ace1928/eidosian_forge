import re
import struct
import zlib
from base64 import b64encode
from datetime import datetime, timedelta
from hashlib import md5
from webob.byterange import ContentRange
from webob.cachecontrol import CacheControl, serialize_cache_control
from webob.compat import (
from webob.cookies import Cookie, make_cookie
from webob.datetime_utils import (
from webob.descriptors import (
from webob.headers import ResponseHeaders
from webob.request import BaseRequest
from webob.util import status_generic_reasons, status_reasons, warn_deprecation
def gzip_app_iter(app_iter):
    size = 0
    crc = zlib.crc32(b'') & 4294967295
    compress = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0)
    yield _gzip_header
    for item in app_iter:
        size += len(item)
        crc = zlib.crc32(item, crc) & 4294967295
        result = compress.compress(item)
        if result:
            yield result
    result = compress.flush()
    if result:
        yield result
    yield struct.pack('<2L', crc, size & 4294967295)