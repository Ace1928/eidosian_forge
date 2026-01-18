import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
def showsyntaxerror(self, filename: Optional[str]=None) -> None:
    """Override the regular handler, the code's copied and pasted from
        code.py, as per showtraceback, but with the syntaxerror callback called
        and the text in a pretty colour."""
    if self.syntaxerror_callback is not None:
        self.syntaxerror_callback()
    exc_type, value, sys.last_traceback = sys.exc_info()
    sys.last_type = exc_type
    sys.last_value = value
    if filename and exc_type is SyntaxError and (value is not None):
        msg = value.args[0]
        args = list(value.args[1])
        if self.bpython_input_re.match(filename):
            args[0] = '<input>'
        value = SyntaxError(msg, tuple(args))
        sys.last_value = value
    exc_formatted = traceback.format_exception_only(exc_type, value)
    self.writetb(exc_formatted)