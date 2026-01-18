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
def list_commands(self, ctx: click.Context) -> List[str]:
    self.maybe_add_run(ctx)
    return super().list_commands(ctx)