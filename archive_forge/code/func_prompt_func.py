import inspect
import io
import itertools
import sys
import typing as t
from gettext import gettext as _
from ._compat import isatty
from ._compat import strip_ansi
from .exceptions import Abort
from .exceptions import UsageError
from .globals import resolve_color_default
from .types import Choice
from .types import convert_type
from .types import ParamType
from .utils import echo
from .utils import LazyFile
def prompt_func(text: str) -> str:
    f = hidden_prompt_func if hide_input else visible_prompt_func
    try:
        echo(text.rstrip(' '), nl=False, err=err)
        return f(' ')
    except (KeyboardInterrupt, EOFError):
        if hide_input:
            echo(None, err=err)
        raise Abort() from None