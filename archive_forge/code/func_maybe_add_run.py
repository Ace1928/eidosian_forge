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
def maybe_add_run(self, ctx: click.Context) -> None:
    maybe_update_state(ctx)
    maybe_add_run_to_cli(self)