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
def get_app_iter(self, environ: WSGIEnvironment) -> t.Iterable[bytes]:
    """Returns the application iterator for the given environ.  Depending
        on the request method and the current status code the return value
        might be an empty response rather than the one from the response.

        If the request method is `HEAD` or the status code is in a range
        where the HTTP specification requires an empty response, an empty
        iterable is returned.

        .. versionadded:: 0.6

        :param environ: the WSGI environment of the request.
        :return: a response iterable.
        """
    status = self.status_code
    if environ['REQUEST_METHOD'] == 'HEAD' or 100 <= status < 200 or status in (204, 304):
        iterable: t.Iterable[bytes] = ()
    elif self.direct_passthrough:
        return self.response
    else:
        iterable = self.iter_encoded()
    return ClosingIterator(iterable, self.close)