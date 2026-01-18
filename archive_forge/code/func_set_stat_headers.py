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
def set_stat_headers(self, stat_result: os.stat_result) -> None:
    content_length = str(stat_result.st_size)
    last_modified = formatdate(stat_result.st_mtime, usegmt=True)
    etag_base = str(stat_result.st_mtime) + '-' + str(stat_result.st_size)
    etag = f'"{md5_hexdigest(etag_base.encode(), usedforsecurity=False)}"'
    self.headers.setdefault('content-length', content_length)
    self.headers.setdefault('last-modified', last_modified)
    self.headers.setdefault('etag', etag)