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
def get_typer_from_module(module: Any) -> Optional[typer.Typer]:
    if state.app:
        obj = getattr(module, state.app, None)
        if not isinstance(obj, typer.Typer):
            typer.echo(f'Not a Typer object: --app {state.app}', err=True)
            sys.exit(1)
        return obj
    if state.func:
        func_obj = getattr(module, state.func, None)
        if not callable(func_obj):
            typer.echo(f'Not a function: --func {state.func}', err=True)
            sys.exit(1)
        sub_app = typer.Typer()
        sub_app.command()(func_obj)
        return sub_app
    local_names = dir(module)
    local_names_set = set(local_names)
    for name in default_app_names:
        if name in local_names_set:
            obj = getattr(module, name, None)
            if isinstance(obj, typer.Typer):
                return obj
    for name in local_names_set - set(default_app_names):
        obj = getattr(module, name)
        if isinstance(obj, typer.Typer):
            return obj
    for func_name in default_func_names:
        func_obj = getattr(module, func_name, None)
        if callable(func_obj):
            sub_app = typer.Typer()
            sub_app.command()(func_obj)
            return sub_app
    for func_name in local_names_set - set(default_func_names):
        func_obj = getattr(module, func_name)
        if callable(func_obj):
            sub_app = typer.Typer()
            sub_app.command()(func_obj)
            return sub_app
    return None