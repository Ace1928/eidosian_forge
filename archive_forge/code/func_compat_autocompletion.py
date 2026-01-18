import errno
import inspect
import os
import sys
from enum import Enum
from gettext import gettext as _
from typing import (
import click
import click.core
import click.formatting
import click.parser
import click.shell_completion
import click.types
import click.utils
def compat_autocompletion(ctx: click.Context, param: click.core.Parameter, incomplete: str) -> List['click.shell_completion.CompletionItem']:
    from click.shell_completion import CompletionItem
    out = []
    for c in autocompletion(ctx, [], incomplete):
        if isinstance(c, tuple):
            use_completion = CompletionItem(c[0], help=c[1])
        else:
            assert isinstance(c, str)
            use_completion = CompletionItem(c)
        if use_completion.value.startswith(incomplete):
            out.append(use_completion)
    return out