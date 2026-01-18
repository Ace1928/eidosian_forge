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
def reevaluate(self, new_code=False):
    """bpython.Repl.undo calls this"""
    if self.watcher:
        self.watcher.reset()
    old_logical_lines = self.history
    old_display_lines = self.display_lines
    self.history = []
    self.display_lines = []
    self.all_logical_lines = []
    if not self.weak_rewind:
        self.interp = self.interp.__class__()
        self.interp.write = self.send_to_stdouterr
        self.coderunner.interp = self.interp
        self.initialize_interp()
    self.buffer = []
    self.display_buffer = []
    self.highlighted_paren = None
    self.process_event(bpythonevents.RunStartupFileEvent())
    self.reevaluating = True
    sys.stdin = ReevaluateFakeStdin(self.stdin, self)
    for line in old_logical_lines:
        self._current_line = line
        self.on_enter(new_code=new_code)
        while self.fake_refresh_requested:
            self.fake_refresh_requested = False
            self.process_event(bpythonevents.RefreshRequestEvent())
    sys.stdin = self.stdin
    self.reevaluating = False
    num_lines_onscreen = len(self.lines_for_display) - max(0, self.scroll_offset)
    display_lines_offscreen = self.display_lines[:len(self.display_lines) - num_lines_onscreen]
    old_display_lines_offscreen = old_display_lines[:len(self.display_lines) - num_lines_onscreen]
    logger.debug('old_display_lines_offscreen %s', '|'.join((str(x) for x in old_display_lines_offscreen)))
    logger.debug('    display_lines_offscreen %s', '|'.join((str(x) for x in display_lines_offscreen)))
    if old_display_lines_offscreen[:len(display_lines_offscreen)] != display_lines_offscreen and (not self.history_already_messed_up):
        self.inconsistent_history = True
    logger.debug('after rewind, self.inconsistent_history is %r', self.inconsistent_history)
    self._cursor_offset = 0
    self.current_line = ''