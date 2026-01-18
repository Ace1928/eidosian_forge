from __future__ import annotations
import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn
from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper
def check_headers(self, headers: Headers) -> None:
    etag = headers.get('etag')
    if etag is not None:
        if etag.startswith(('W/', 'w/')):
            if etag.startswith('w/'):
                warn('Weak etag indicator should be upper case.', HTTPWarning, stacklevel=4)
            etag = etag[2:]
        if not etag[:1] == etag[-1:] == '"':
            warn('Unquoted etag emitted.', HTTPWarning, stacklevel=4)
    location = headers.get('location')
    if location is not None:
        if not urlparse(location).netloc:
            warn('Absolute URLs required for location header.', HTTPWarning, stacklevel=4)