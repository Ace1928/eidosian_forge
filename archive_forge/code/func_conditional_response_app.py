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
def conditional_response_app(self, environ, start_response):
    """
        Like the normal ``__call__`` interface, but checks conditional headers:

            * ``If-Modified-Since``   (``304 Not Modified``; only on ``GET``,
              ``HEAD``)
            * ``If-None-Match``       (``304 Not Modified``; only on ``GET``,
              ``HEAD``)
            * ``Range``               (``406 Partial Content``; only on ``GET``,
              ``HEAD``)
        """
    req = BaseRequest(environ)
    headerlist = self._abs_headerlist(environ)
    method = environ.get('REQUEST_METHOD', 'GET')
    if method in self._safe_methods:
        status304 = False
        if req.if_none_match and self.etag:
            status304 = self.etag in req.if_none_match
        elif req.if_modified_since and self.last_modified:
            status304 = self.last_modified <= req.if_modified_since
        if status304:
            start_response('304 Not Modified', filter_headers(headerlist))
            return EmptyResponse(self._app_iter)
    if req.range and self in req.if_range and (self.content_range is None) and (method in ('HEAD', 'GET')) and (self.status_code == 200) and (self.content_length is not None):
        content_range = req.range.content_range(self.content_length)
        if content_range is None:
            iter_close(self._app_iter)
            body = bytes_('Requested range not satisfiable: %s' % req.range)
            headerlist = [('Content-Length', str(len(body))), ('Content-Range', str(ContentRange(None, None, self.content_length))), ('Content-Type', 'text/plain')] + filter_headers(headerlist)
            start_response('416 Requested Range Not Satisfiable', headerlist)
            if method == 'HEAD':
                return ()
            return [body]
        else:
            app_iter = self.app_iter_range(content_range.start, content_range.stop)
            if app_iter is not None:
                assert content_range.start is not None
                headerlist = [('Content-Length', str(content_range.stop - content_range.start)), ('Content-Range', str(content_range))] + filter_headers(headerlist, ('content-length',))
                start_response('206 Partial Content', headerlist)
                if method == 'HEAD':
                    return EmptyResponse(app_iter)
                return app_iter
    start_response(self.status, headerlist)
    if method == 'HEAD':
        return EmptyResponse(self._app_iter)
    return self._app_iter