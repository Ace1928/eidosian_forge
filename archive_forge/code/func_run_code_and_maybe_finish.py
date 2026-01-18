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
def run_code_and_maybe_finish(self, for_code=None):
    r = self.coderunner.run_code(for_code=for_code)
    if r:
        logger.debug('----- Running finish command stuff -----')
        logger.debug('saved_indent: %r', self.saved_indent)
        err = self.saved_predicted_parse_error
        self.saved_predicted_parse_error = False
        indent = self.saved_indent
        if err:
            indent = 0
        if self.rl_history.index == 0:
            self._set_current_line(' ' * indent, update_completion=True)
        else:
            self._set_current_line(self.rl_history.entries[-self.rl_history.index], reset_rl_history=False)
        self.cursor_offset = len(self.current_line)