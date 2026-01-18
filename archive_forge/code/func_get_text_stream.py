import os
import re
import sys
import typing as t
from functools import update_wrapper
from types import ModuleType
from types import TracebackType
from ._compat import _default_text_stderr
from ._compat import _default_text_stdout
from ._compat import _find_binary_writer
from ._compat import auto_wrap_for_ansi
from ._compat import binary_streams
from ._compat import open_stream
from ._compat import should_strip_ansi
from ._compat import strip_ansi
from ._compat import text_streams
from ._compat import WIN
from .globals import resolve_color_default
def get_text_stream(name: "te.Literal['stdin', 'stdout', 'stderr']", encoding: t.Optional[str]=None, errors: t.Optional[str]='strict') -> t.TextIO:
    """Returns a system stream for text processing.  This usually returns
    a wrapped stream around a binary stream returned from
    :func:`get_binary_stream` but it also can take shortcuts for already
    correctly configured streams.

    :param name: the name of the stream to open.  Valid names are ``'stdin'``,
                 ``'stdout'`` and ``'stderr'``
    :param encoding: overrides the detected default encoding.
    :param errors: overrides the default error mode.
    """
    opener = text_streams.get(name)
    if opener is None:
        raise TypeError(f"Unknown standard stream '{name}'")
    return opener(encoding, errors)