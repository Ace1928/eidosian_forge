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
def call_on_close(self, func: t.Callable[[], t.Any]) -> t.Callable[[], t.Any]:
    """Adds a function to the internal list of functions that should
        be called as part of closing down the response.  Since 0.7 this
        function also returns the function that was passed so that this
        can be used as a decorator.

        .. versionadded:: 0.6
        """
    self._on_close.append(func)
    return func