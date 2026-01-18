from __future__ import annotations
import inspect
import warnings
from json import dumps as json_dumps
from typing import (
from urllib.parse import urlencode
from ._exceptions import StreamClosed, StreamConsumed
from ._multipart import MultipartStream
from ._types import (
from ._utils import peek_filelike_length, primitive_value_to_str
def encode_json(json: Any) -> tuple[dict[str, str], ByteStream]:
    body = json_dumps(json).encode('utf-8')
    content_length = str(len(body))
    content_type = 'application/json'
    headers = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))