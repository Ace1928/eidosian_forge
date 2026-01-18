from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
def replace_query_params(self, **kwargs: typing.Any) -> URL:
    query = urlencode([(str(key), str(value)) for key, value in kwargs.items()])
    return self.replace(query=query)