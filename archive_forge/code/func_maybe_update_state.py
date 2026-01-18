import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, List, Optional
import click
import typer
import typer.core
from click import Command, Group, Option
from . import __version__
def maybe_update_state(ctx: click.Context) -> None:
    path_or_module = ctx.params.get('path_or_module')
    if path_or_module:
        file_path = Path(path_or_module)
        if file_path.exists() and file_path.is_file():
            state.file = file_path
        else:
            if not re.fullmatch('[a-zA-Z_]\\w*(\\.[a-zA-Z_]\\w*)*', path_or_module):
                typer.echo(f'Not a valid file or Python module: {path_or_module}', err=True)
                sys.exit(1)
            state.module = path_or_module
    app_name = ctx.params.get('app')
    if app_name:
        state.app = app_name
    func_name = ctx.params.get('func')
    if func_name:
        state.func = func_name