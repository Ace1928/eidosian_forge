from __future__ import annotations
import http.cookies
import json
import os
import stat
import typing
import warnings
from datetime import datetime
from email.utils import format_datetime, formatdate
from functools import partial
from mimetypes import guess_type
from urllib.parse import quote
import anyio
import anyio.to_thread
from starlette._compat import md5_hexdigest
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, MutableHeaders
from starlette.types import Receive, Scope, Send
def init_headers(self, headers: typing.Mapping[str, str] | None=None) -> None:
    if headers is None:
        raw_headers: list[tuple[bytes, bytes]] = []
        populate_content_length = True
        populate_content_type = True
    else:
        raw_headers = [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in headers.items()]
        keys = [h[0] for h in raw_headers]
        populate_content_length = b'content-length' not in keys
        populate_content_type = b'content-type' not in keys
    body = getattr(self, 'body', None)
    if body is not None and populate_content_length and (not (self.status_code < 200 or self.status_code in (204, 304))):
        content_length = str(len(body))
        raw_headers.append((b'content-length', content_length.encode('latin-1')))
    content_type = self.media_type
    if content_type is not None and populate_content_type:
        if content_type.startswith('text/') and 'charset=' not in content_type.lower():
            content_type += '; charset=' + self.charset
        raw_headers.append((b'content-type', content_type.encode('latin-1')))
    self.raw_headers = raw_headers