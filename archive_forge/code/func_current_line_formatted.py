import contextlib
import errno
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from enum import Enum
from types import FrameType, TracebackType
from typing import (
from .._typing_compat import Literal
import greenlet
from curtsies import (
from curtsies.configfile_keynames import keymap as key_dispatch
from curtsies.input import is_main_thread
from curtsies.window import CursorAwareWindow
from cwcwidth import wcswidth
from pygments import format as pygformat
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from . import events as bpythonevents, sitefix, replpainter as paint
from ..config import Config
from .coderunner import (
from .filewatch import ModuleChangedEventHandler
from .interaction import StatusBar
from .interpreter import (
from .manual_readline import (
from .parse import parse as bpythonparse, func_for_letter, color_for_letter
from .preprocess import preprocess
from .. import __version__
from ..config import getpreferredencoding
from ..formatter import BPythonFormatter
from ..pager import get_pager_command
from ..repl import (
from ..translations import _
from ..line import CHARACTER_PAIR_MAP
@property
def current_line_formatted(self):
    """The colored current line (no prompt, not wrapped)"""
    if self.config.syntax:
        fs = bpythonparse(pygformat(self.tokenize(self.current_line), self.formatter))
        if self.incr_search_mode != SearchMode.NO_SEARCH:
            if self.incr_search_target in self.current_line:
                fs = fmtfuncs.on_magenta(self.incr_search_target).join(fs.split(self.incr_search_target))
        elif self.rl_history.saved_line and self.rl_history.saved_line in self.current_line:
            if self.config.curtsies_right_arrow_completion and self.rl_history.index != 0:
                fs = fmtfuncs.on_magenta(self.rl_history.saved_line).join(fs.split(self.rl_history.saved_line))
        logger.debug('Display line %r -> %r', self.current_line, fs)
    else:
        fs = fmtstr(self.current_line)
    if hasattr(self, 'old_fs') and str(fs) != str(self.old_fs):
        pass
    self.old_fs = fs
    return fs