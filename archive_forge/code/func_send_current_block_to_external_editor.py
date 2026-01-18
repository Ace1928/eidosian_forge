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
def send_current_block_to_external_editor(self, filename=None):
    """
        Sends the current code block to external editor to be edited. Usually bound to C-x.
        """
    text = self.send_to_external_editor(self.get_current_block())
    lines = [line for line in text.split('\n')]
    while lines and (not lines[-1].split()):
        lines.pop()
    events = '\n'.join(lines + ([''] if len(lines) == 1 else ['', '']))
    self.clear_current_block()
    with self.in_paste_mode():
        for e in events:
            self.process_simple_keypress(e)
    self.cursor_offset = len(self.current_line)