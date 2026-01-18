import enum
import errno
import inspect
import os
import sys
import typing as t
from collections import abc
from contextlib import contextmanager
from contextlib import ExitStack
from functools import update_wrapper
from gettext import gettext as _
from gettext import ngettext
from itertools import repeat
from types import TracebackType
from . import types
from .exceptions import Abort
from .exceptions import BadParameter
from .exceptions import ClickException
from .exceptions import Exit
from .exceptions import MissingParameter
from .exceptions import UsageError
from .formatting import HelpFormatter
from .formatting import join_options
from .globals import pop_context
from .globals import push_context
from .parser import _flag_needs_value
from .parser import OptionParser
from .parser import split_opt
from .termui import confirm
from .termui import prompt
from .termui import style
from .utils import _detect_program_name
from .utils import _expand_args
from .utils import echo
from .utils import make_default_short_help
from .utils import make_str
from .utils import PacifyFlushWrapper
def get_help_option(self, ctx: Context) -> t.Optional['Option']:
    """Returns the help option object."""
    help_options = self.get_help_option_names(ctx)
    if not help_options or not self.add_help_option:
        return None

    def show_help(ctx: Context, param: 'Parameter', value: str) -> None:
        if value and (not ctx.resilient_parsing):
            echo(ctx.get_help(), color=ctx.color)
            ctx.exit()
    return Option(help_options, is_flag=True, is_eager=True, expose_value=False, callback=show_help, help=_('Show this message and exit.'))