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
def print_response(response: Response) -> None:
    console = rich.console.Console()
    lexer_name = get_lexer_for_response(response)
    if lexer_name:
        if lexer_name.lower() == 'json':
            try:
                data = response.json()
                text = json.dumps(data, indent=4)
            except ValueError:
                text = response.text
        else:
            text = response.text
        syntax = rich.syntax.Syntax(text, lexer_name, theme='ansi_dark', word_wrap=True)
        console.print(syntax)
    else:
        console.print(f'<{len(response.content)} bytes of binary data>')