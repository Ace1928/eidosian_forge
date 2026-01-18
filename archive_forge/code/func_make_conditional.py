from __future__ import annotations
import json
import typing as t
from http import HTTPStatus
from urllib.parse import urljoin
from ..datastructures import Headers
from ..http import remove_entity_headers
from ..sansio.response import Response as _SansIOResponse
from ..urls import _invalid_iri_to_uri
from ..urls import iri_to_uri
from ..utils import cached_property
from ..wsgi import ClosingIterator
from ..wsgi import get_current_url
from werkzeug._internal import _get_environ
from werkzeug.http import generate_etag
from werkzeug.http import http_date
from werkzeug.http import is_resource_modified
from werkzeug.http import parse_etags
from werkzeug.http import parse_range_header
from werkzeug.wsgi import _RangeWrapper
def make_conditional(self, request_or_environ: WSGIEnvironment | Request, accept_ranges: bool | str=False, complete_length: int | None=None) -> Response:
    """Make the response conditional to the request.  This method works
        best if an etag was defined for the response already.  The `add_etag`
        method can be used to do that.  If called without etag just the date
        header is set.

        This does nothing if the request method in the request or environ is
        anything but GET or HEAD.

        For optimal performance when handling range requests, it's recommended
        that your response data object implements `seekable`, `seek` and `tell`
        methods as described by :py:class:`io.IOBase`.  Objects returned by
        :meth:`~werkzeug.wsgi.wrap_file` automatically implement those methods.

        It does not remove the body of the response because that's something
        the :meth:`__call__` function does for us automatically.

        Returns self so that you can do ``return resp.make_conditional(req)``
        but modifies the object in-place.

        :param request_or_environ: a request object or WSGI environment to be
                                   used to make the response conditional
                                   against.
        :param accept_ranges: This parameter dictates the value of
                              `Accept-Ranges` header. If ``False`` (default),
                              the header is not set. If ``True``, it will be set
                              to ``"bytes"``. If it's a string, it will use this
                              value.
        :param complete_length: Will be used only in valid Range Requests.
                                It will set `Content-Range` complete length
                                value and compute `Content-Length` real value.
                                This parameter is mandatory for successful
                                Range Requests completion.
        :raises: :class:`~werkzeug.exceptions.RequestedRangeNotSatisfiable`
                 if `Range` header could not be parsed or satisfied.

        .. versionchanged:: 2.0
            Range processing is skipped if length is 0 instead of
            raising a 416 Range Not Satisfiable error.
        """
    environ = _get_environ(request_or_environ)
    if environ['REQUEST_METHOD'] in ('GET', 'HEAD'):
        if 'date' not in self.headers:
            self.headers['Date'] = http_date()
        is206 = self._process_range_request(environ, complete_length, accept_ranges)
        if not is206 and (not is_resource_modified(environ, self.headers.get('etag'), None, self.headers.get('last-modified'))):
            if parse_etags(environ.get('HTTP_IF_MATCH')):
                self.status_code = 412
            else:
                self.status_code = 304
        if self.automatically_set_content_length and 'content-length' not in self.headers:
            length = self.calculate_content_length()
            if length is not None:
                self.headers['Content-Length'] = str(length)
    return self