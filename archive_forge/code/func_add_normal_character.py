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
def add_normal_character(self, char, narrow_search=True):
    if len(char) > 1 or is_nop(char):
        return
    if self.incr_search_mode != SearchMode.NO_SEARCH:
        self.add_to_incremental_search(char)
    else:
        self._set_current_line(self.current_line[:self.cursor_offset] + char + self.current_line[self.cursor_offset:], update_completion=False, reset_rl_history=False, clear_special_mode=False)
        if narrow_search:
            self.cursor_offset += 1
        else:
            self._cursor_offset += 1
    if self.config.cli_trim_prompts and self.current_line.startswith(self.ps1):
        self.current_line = self.current_line[4:]
        if narrow_search:
            self.cursor_offset = max(0, self.cursor_offset - 4)
        else:
            self._cursor_offset += max(0, self.cursor_offset - 4)