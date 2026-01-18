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
def make_step(self, n_steps: int) -> None:
    self.pos += n_steps
    if self.length is not None and self.pos >= self.length:
        self.finished = True
    if time.time() - self.last_eta < 1.0:
        return
    self.last_eta = time.time()
    if self.pos:
        step = (time.time() - self.start) / self.pos
    else:
        step = time.time() - self.start
    self.avg = self.avg[-6:] + [step]
    self.eta_known = self.length is not None