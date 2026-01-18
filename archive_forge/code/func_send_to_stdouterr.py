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
def send_to_stdouterr(self, output):
    """Send unicode strings or FmtStr to Repl stdout or stderr

        Must be able to handle FmtStrs because interpreter pass in
        tracebacks already formatted."""
    lines = output.split('\n')
    logger.debug('display_lines: %r', self.display_lines)
    self.current_stdouterr_line += lines[0]
    if len(lines) > 1:
        self.display_lines.extend(paint.display_linize(self.current_stdouterr_line, self.width, blank_line=True))
        self.display_lines.extend(sum((paint.display_linize(line, self.width, blank_line=True) for line in lines[1:-1]), []))
        for line in itertools.chain((self.current_stdouterr_line,), lines[1:-1]):
            if isinstance(line, FmtStr):
                self.all_logical_lines.append((line.s, LineType.OUTPUT))
            else:
                self.all_logical_lines.append((line, LineType.OUTPUT))
        self.current_stdouterr_line = lines[-1]
    logger.debug('display_lines: %r', self.display_lines)