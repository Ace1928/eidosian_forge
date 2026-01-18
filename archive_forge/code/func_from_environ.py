from __future__ import annotations
import dataclasses
import mimetypes
import sys
import typing as t
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from itertools import chain
from random import random
from tempfile import TemporaryFile
from time import time
from urllib.parse import unquote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from ._internal import _get_environ
from ._internal import _wsgi_decoding_dance
from ._internal import _wsgi_encoding_dance
from .datastructures import Authorization
from .datastructures import CallbackDict
from .datastructures import CombinedMultiDict
from .datastructures import EnvironHeaders
from .datastructures import FileMultiDict
from .datastructures import Headers
from .datastructures import MultiDict
from .http import dump_cookie
from .http import dump_options_header
from .http import parse_cookie
from .http import parse_date
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartEncoder
from .sansio.multipart import Preamble
from .urls import _urlencode
from .urls import iri_to_uri
from .utils import cached_property
from .utils import get_content_type
from .wrappers.request import Request
from .wrappers.response import Response
from .wsgi import ClosingIterator
from .wsgi import get_current_url
@classmethod
def from_environ(cls, environ: WSGIEnvironment, **kwargs: t.Any) -> EnvironBuilder:
    """Turn an environ dict back into a builder. Any extra kwargs
        override the args extracted from the environ.

        .. versionchanged:: 2.0
            Path and query values are passed through the WSGI decoding
            dance to avoid double encoding.

        .. versionadded:: 0.15
        """
    headers = Headers(EnvironHeaders(environ))
    out = {'path': _wsgi_decoding_dance(environ['PATH_INFO']), 'base_url': cls._make_base_url(environ['wsgi.url_scheme'], headers.pop('Host'), _wsgi_decoding_dance(environ['SCRIPT_NAME'])), 'query_string': _wsgi_decoding_dance(environ['QUERY_STRING']), 'method': environ['REQUEST_METHOD'], 'input_stream': environ['wsgi.input'], 'content_type': headers.pop('Content-Type', None), 'content_length': headers.pop('Content-Length', None), 'errors_stream': environ['wsgi.errors'], 'multithread': environ['wsgi.multithread'], 'multiprocess': environ['wsgi.multiprocess'], 'run_once': environ['wsgi.run_once'], 'headers': headers}
    out.update(kwargs)
    return cls(**out)