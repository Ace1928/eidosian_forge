from __future__ import annotations
import typing as t
from io import BytesIO
from urllib.parse import parse_qsl
from ._internal import _plain_int
from .datastructures import FileStorage
from .datastructures import Headers
from .datastructures import MultiDict
from .exceptions import RequestEntityTooLarge
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartDecoder
from .sansio.multipart import NeedData
from .wsgi import get_content_length
from .wsgi import get_input_stream
def get_part_charset(self, headers: Headers) -> str:
    content_type = headers.get('content-type')
    if content_type:
        parameters = parse_options_header(content_type)[1]
        ct_charset = parameters.get('charset', '').lower()
        if ct_charset in {'ascii', 'us-ascii', 'utf-8', 'iso-8859-1'}:
            return ct_charset
    return 'utf-8'