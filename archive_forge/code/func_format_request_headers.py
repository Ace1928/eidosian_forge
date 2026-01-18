from __future__ import annotations
import functools
import json
import sys
import typing
import click
import httpcore
import pygments.lexers
import pygments.util
import rich.console
import rich.markup
import rich.progress
import rich.syntax
import rich.table
from ._client import Client
from ._exceptions import RequestError
from ._models import Response
from ._status_codes import codes
def format_request_headers(request: httpcore.Request, http2: bool=False) -> str:
    version = 'HTTP/2' if http2 else 'HTTP/1.1'
    headers = [(name.lower() if http2 else name, value) for name, value in request.headers]
    method = request.method.decode('ascii')
    target = request.url.target.decode('ascii')
    lines = [f'{method} {target} {version}'] + [f'{name.decode('ascii')}: {value.decode('ascii')}' for name, value in headers]
    return '\n'.join(lines)