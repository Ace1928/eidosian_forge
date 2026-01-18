import curses
import errno
import functools
import math
import os
import platform
import re
import struct
import sys
import time
from typing import (
from ._typing_compat import Literal
import unicodedata
from dataclasses import dataclass
from pygments import format
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from .formatter import BPythonFormatter
from .config import getpreferredencoding, Config
from .keys import cli_key_dispatch as key_dispatch
from . import translations
from .translations import _
from . import repl, inspection
from . import args as bpargs
from .pager import page
from .args import parse as argsparse
def init_wins(scr: '_CursesWindow', config: Config) -> Tuple['_CursesWindow', Statusbar]:
    """Initialise the two windows (the main repl interface and the little
    status bar at the bottom with some stuff in it)"""
    background = get_colpair(config, 'background')
    h, w = gethw()
    main_win = newwin(background, h - 1, w, 0, 0)
    main_win.scrollok(True)
    main_win.keypad(1)
    commands = ((_('Rewind'), config.undo_key), (_('Save'), config.save_key), (_('Pastebin'), config.pastebin_key), (_('Pager'), config.last_output_key), (_('Show Source'), config.show_source_key))
    message = '  '.join((f'<{key}> {command}' for command, key in commands if key))
    statusbar = Statusbar(scr, main_win, background, config, message, get_colpair(config, 'main'))
    return (main_win, statusbar)