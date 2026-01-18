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
def substitute_cseq(self) -> Tuple[int, str]:
    """Returns a new line by substituting a common sequence in, and update
        matches"""
    assert self.completer is not None
    cseq = os.path.commonprefix(self.matches)
    new_cursor_offset, new_line = self.substitute(cseq)
    if len(self.matches) == 1:
        self.clear()
    else:
        self.update(new_cursor_offset, new_line, self.matches, self.completer)
        if len(self.matches) == 1:
            self.clear()
    return (new_cursor_offset, new_line)