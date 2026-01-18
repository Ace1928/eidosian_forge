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
def format_response_headers(http_version: bytes, status: int, reason_phrase: bytes | None, headers: list[tuple[bytes, bytes]]) -> str:
    version = http_version.decode('ascii')
    reason = codes.get_reason_phrase(status) if reason_phrase is None else reason_phrase.decode('ascii')
    lines = [f'{version} {status} {reason}'] + [f'{name.decode('ascii')}: {value.decode('ascii')}' for name, value in headers]
    return '\n'.join(lines)