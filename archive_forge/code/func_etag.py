import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
from concurrent.futures import Executor
from http import HTTPStatus
from http.cookies import SimpleCookie
from typing import (
from multidict import CIMultiDict, istr
from . import hdrs, payload
from .abc import AbstractStreamWriter
from .compression_utils import ZLibCompressor
from .helpers import (
from .http import SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONEncoder, LooseHeaders
@etag.setter
def etag(self, value: Optional[Union[ETag, str]]) -> None:
    if value is None:
        self._headers.pop(hdrs.ETAG, None)
    elif isinstance(value, str) and value == ETAG_ANY or (isinstance(value, ETag) and value.value == ETAG_ANY):
        self._headers[hdrs.ETAG] = ETAG_ANY
    elif isinstance(value, str):
        validate_etag_value(value)
        self._headers[hdrs.ETAG] = f'"{value}"'
    elif isinstance(value, ETag) and isinstance(value.value, str):
        validate_etag_value(value.value)
        hdr_value = f'W/"{value.value}"' if value.is_weak else f'"{value.value}"'
        self._headers[hdrs.ETAG] = hdr_value
    else:
        raise ValueError(f'Unsupported etag type: {type(value)}. etag must be str, ETag or None')