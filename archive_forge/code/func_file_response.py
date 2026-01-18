from __future__ import annotations
import importlib.util
import os
import stat
import typing
from email.utils import parsedate
import anyio
import anyio.to_thread
from starlette._utils import get_route_path
from starlette.datastructures import URL, Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.types import Receive, Scope, Send
def file_response(self, full_path: PathLike, stat_result: os.stat_result, scope: Scope, status_code: int=200) -> Response:
    request_headers = Headers(scope=scope)
    response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
    if self.is_not_modified(response.headers, request_headers):
        return NotModifiedResponse(response.headers)
    return response