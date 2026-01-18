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
def md5_etag(self, body=None, set_content_md5=False):
    """
        Generate an etag for the response object using an MD5 hash of
        the body (the ``body`` parameter, or ``self.body`` if not given).

        Sets ``self.etag``.

        If ``set_content_md5`` is ``True``, sets ``self.content_md5`` as well.
        """
    if body is None:
        body = self.body
    md5_digest = md5(body).digest()
    md5_digest = b64encode(md5_digest)
    md5_digest = md5_digest.replace(b'\n', b'')
    md5_digest = native_(md5_digest)
    self.etag = md5_digest.strip('=')
    if set_content_md5:
        self.content_md5 = md5_digest