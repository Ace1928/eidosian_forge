import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from hashlib import md5, sha1, sha256
from http.cookies import CookieError, Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .helpers import (
from .http import (
from .log import client_logger
from .streams import StreamReader
from .typedefs import (
@reify
def content_disposition(self) -> Optional[ContentDisposition]:
    raw = self._headers.get(hdrs.CONTENT_DISPOSITION)
    if raw is None:
        return None
    disposition_type, params_dct = multipart.parse_content_disposition(raw)
    params = MappingProxyType(params_dct)
    filename = multipart.content_disposition_filename(params)
    return ContentDisposition(disposition_type, params, filename)