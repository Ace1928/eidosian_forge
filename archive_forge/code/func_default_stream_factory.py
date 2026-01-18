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
def default_stream_factory(total_content_length: int | None, content_type: str | None, filename: str | None, content_length: int | None=None) -> t.IO[bytes]:
    max_size = 1024 * 500
    if SpooledTemporaryFile is not None:
        return t.cast(t.IO[bytes], SpooledTemporaryFile(max_size=max_size, mode='rb+'))
    elif total_content_length is None or total_content_length > max_size:
        return t.cast(t.IO[bytes], TemporaryFile('rb+'))
    return BytesIO()