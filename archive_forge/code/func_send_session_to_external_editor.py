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
def send_session_to_external_editor(self, filename=None):
    """
        Sends entire bpython session to external editor to be edited. Usually bound to F7.
        """
    for_editor = EDIT_SESSION_HEADER
    for_editor += self.get_session_formatted_for_file()
    text = self.send_to_external_editor(for_editor)
    if text == for_editor:
        self.status_bar.message(_('Session not reevaluated because it was not edited'))
        return
    lines = text.split('\n')
    if len(lines) and (not lines[-1].strip()):
        lines.pop()
    if len(lines) and lines[-1].startswith('### '):
        current_line = lines[-1][4:]
    else:
        current_line = ''
    from_editor = [line for line in lines if line[:6] != '# OUT:' and line[:3] != '###']
    if all((not line.strip() for line in from_editor)):
        self.status_bar.message(_('Session not reevaluated because saved file was blank'))
        return
    source = preprocess('\n'.join(from_editor), self.interp.compile)
    lines = source.split('\n')
    self.history = lines
    self.reevaluate(new_code=True)
    self.current_line = current_line
    self.cursor_offset = len(self.current_line)
    self.status_bar.message(_('Session edited and reevaluated'))