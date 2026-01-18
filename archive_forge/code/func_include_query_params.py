from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
def include_query_params(self, **kwargs: typing.Any) -> URL:
    params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
    params.update({str(key): str(value) for key, value in kwargs.items()})
    query = urlencode(params.multi_items())
    return self.replace(query=query)