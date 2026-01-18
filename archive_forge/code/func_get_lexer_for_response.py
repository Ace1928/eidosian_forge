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
def get_lexer_for_response(response: Response) -> str:
    content_type = response.headers.get('Content-Type')
    if content_type is not None:
        mime_type, _, _ = content_type.partition(';')
        try:
            return typing.cast(str, pygments.lexers.get_lexer_for_mimetype(mime_type.strip()).name)
        except pygments.util.ClassNotFound:
            pass
    return ''