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
def raw_terminal() -> t.ContextManager[int]:
    from ._termui_impl import raw_terminal as f
    return f()