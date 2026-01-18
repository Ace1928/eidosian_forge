import inspect
import sys
from collections import defaultdict
from gettext import gettext as _
from os import getenv
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Union
import click
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, RenderableType, group
from rich.emoji import Emoji
from rich.highlighter import RegexHighlighter
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
def rich_format_error(self: click.ClickException) -> None:
    """Print richly formatted click errors.

    Called by custom exception handler to print richly formatted click errors.
    Mimics original click.ClickException.echo() function but with rich formatting.
    """
    console = _get_rich_console(stderr=True)
    ctx: Union[click.Context, None] = getattr(self, 'ctx', None)
    if ctx is not None:
        console.print(ctx.get_usage())
    if ctx is not None and ctx.command.get_help_option(ctx) is not None:
        console.print(f"Try [blue]'{ctx.command_path} {ctx.help_option_names[0]}'[/] for help.", style=STYLE_ERRORS_SUGGESTION)
    console.print(Panel(highlighter(self.format_message()), border_style=STYLE_ERRORS_PANEL_BORDER, title=ERRORS_PANEL_TITLE, title_align=ALIGN_ERRORS_PANEL))