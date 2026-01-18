import time of Click down, some infrequently used functionality is
import contextlib
import math
import os
import sys
import time
import typing as t
from gettext import gettext as _
from io import StringIO
from types import TracebackType
from ._compat import _default_text_stdout
from ._compat import CYGWIN
from ._compat import get_best_encoding
from ._compat import isatty
from ._compat import open_stream
from ._compat import strip_ansi
from ._compat import term_len
from ._compat import WIN
from .exceptions import ClickException
from .utils import echo
def format_bar(self) -> str:
    if self.length is not None:
        bar_length = int(self.pct * self.width)
        bar = self.fill_char * bar_length
        bar += self.empty_char * (self.width - bar_length)
    elif self.finished:
        bar = self.fill_char * self.width
    else:
        chars = list(self.empty_char * (self.width or 1))
        if self.time_per_iteration != 0:
            chars[int((math.cos(self.pos * self.time_per_iteration) / 2.0 + 0.5) * self.width)] = self.fill_char
        bar = ''.join(chars)
    return bar