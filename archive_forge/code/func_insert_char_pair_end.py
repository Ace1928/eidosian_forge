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
def insert_char_pair_end(self, e):
    """Accepts character which is a part of CHARACTER_PAIR_MAP
        like brackets and quotes, and checks whether it should be
        inserted to the line or overwritten

        e.x. if you type ")" (rparen) , and your cursor is directly
        above another ")" (rparen) in the cmd, this will just skip
        it and move the cursor.
        If there is no same character underneath the cursor, the
        character will be printed/appended to the line
        """
    if self.config.brackets_completion:
        if self.cursor_offset < len(self._current_line):
            if self._current_line[self.cursor_offset] == e:
                self.cursor_offset += 1
                return
    self.add_normal_character(e)