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
def decode_content(self):
    content_encoding = self.content_encoding or 'identity'
    if content_encoding == 'identity':
        return
    if content_encoding not in ('gzip', 'deflate'):
        raise ValueError("I don't know how to decode the content %s" % content_encoding)
    if content_encoding == 'gzip':
        from gzip import GzipFile
        from io import BytesIO
        gzip_f = GzipFile(filename='', mode='r', fileobj=BytesIO(self.body))
        self.body = gzip_f.read()
        self.content_encoding = None
        gzip_f.close()
    else:
        try:
            self.body = zlib.decompress(self.body)
        except zlib.error:
            self.body = zlib.decompress(self.body, -15)
        self.content_encoding = None