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
def add_to_incremental_search(self, char=None, backspace=False):
    """Modify the current search term while in incremental search.

        The only operations allowed in incremental search mode are
        adding characters and backspacing."""
    if backspace:
        self.incr_search_target = self.incr_search_target[:-1]
    elif char is not None:
        self.incr_search_target += char
    else:
        raise ValueError('must provide a char or set backspace to True')
    if self.incr_search_mode == SearchMode.REVERSE_INCREMENTAL_SEARCH:
        self.incremental_search(reverse=True, include_current=True)
    elif self.incr_search_mode == SearchMode.INCREMENTAL_SEARCH:
        self.incremental_search(include_current=True)
    else:
        raise ValueError('add_to_incremental_search not in a special mode')