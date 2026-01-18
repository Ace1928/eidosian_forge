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
def get_environ(self) -> WSGIEnvironment:
    """Return the built environ.

        .. versionchanged:: 0.15
            The content type and length headers are set based on
            input stream detection. Previously this only set the WSGI
            keys.
        """
    input_stream = self.input_stream
    content_length = self.content_length
    mimetype = self.mimetype
    content_type = self.content_type
    if input_stream is not None:
        start_pos = input_stream.tell()
        input_stream.seek(0, 2)
        end_pos = input_stream.tell()
        input_stream.seek(start_pos)
        content_length = end_pos - start_pos
    elif mimetype == 'multipart/form-data':
        input_stream, content_length, boundary = stream_encode_multipart(CombinedMultiDict([self.form, self.files]))
        content_type = f'{mimetype}; boundary="{boundary}"'
    elif mimetype == 'application/x-www-form-urlencoded':
        form_encoded = _urlencode(self.form).encode('ascii')
        content_length = len(form_encoded)
        input_stream = BytesIO(form_encoded)
    else:
        input_stream = BytesIO()
    result: WSGIEnvironment = {}
    if self.environ_base:
        result.update(self.environ_base)

    def _path_encode(x: str) -> str:
        return _wsgi_encoding_dance(unquote(x))
    raw_uri = _wsgi_encoding_dance(self.request_uri)
    result.update({'REQUEST_METHOD': self.method, 'SCRIPT_NAME': _path_encode(self.script_root), 'PATH_INFO': _path_encode(self.path), 'QUERY_STRING': _wsgi_encoding_dance(self.query_string), 'REQUEST_URI': raw_uri, 'RAW_URI': raw_uri, 'SERVER_NAME': self.server_name, 'SERVER_PORT': str(self.server_port), 'HTTP_HOST': self.host, 'SERVER_PROTOCOL': self.server_protocol, 'wsgi.version': self.wsgi_version, 'wsgi.url_scheme': self.url_scheme, 'wsgi.input': input_stream, 'wsgi.errors': self.errors_stream, 'wsgi.multithread': self.multithread, 'wsgi.multiprocess': self.multiprocess, 'wsgi.run_once': self.run_once})
    headers = self.headers.copy()
    headers.remove('Content-Type')
    headers.remove('Content-Length')
    if content_type is not None:
        result['CONTENT_TYPE'] = content_type
    if content_length is not None:
        result['CONTENT_LENGTH'] = str(content_length)
    combined_headers = defaultdict(list)
    for key, value in headers.to_wsgi_list():
        combined_headers[f'HTTP_{key.upper().replace('-', '_')}'].append(value)
    for key, values in combined_headers.items():
        result[key] = ', '.join(values)
    if self.environ_overrides:
        result.update(self.environ_overrides)
    return result