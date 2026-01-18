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
def process_key_event(self, e: str) -> None:
    if self.status_bar.has_focus:
        return self.status_bar.process_event(e)
    if self.stdin.has_focus:
        return self.stdin.process_event(e)
    if e in key_dispatch[self.config.right_key] + key_dispatch[self.config.end_of_line_key] + ('<RIGHT>',) and self.config.curtsies_right_arrow_completion and (self.cursor_offset == len(self.current_line)):
        self.current_line += self.current_suggestion
        self.cursor_offset = len(self.current_line)
    elif e in ('<UP>',) + key_dispatch[self.config.up_one_line_key]:
        self.up_one_line()
    elif e in ('<DOWN>',) + key_dispatch[self.config.down_one_line_key]:
        self.down_one_line()
    elif e == '<Ctrl-d>':
        self.on_control_d()
    elif e == '<Ctrl-o>':
        self.operate_and_get_next()
    elif e == '<Esc+.>':
        self.get_last_word()
    elif e in key_dispatch[self.config.reverse_incremental_search_key]:
        self.incremental_search(reverse=True)
    elif e in key_dispatch[self.config.incremental_search_key]:
        self.incremental_search()
    elif e in ('<BACKSPACE>',) + key_dispatch[self.config.backspace_key] and self.incr_search_mode != SearchMode.NO_SEARCH:
        self.add_to_incremental_search(self, backspace=True)
    elif e in self.edit_keys.cut_buffer_edits:
        self.readline_kill(e)
    elif e in self.edit_keys.simple_edits:
        self.cursor_offset, self.current_line = self.edit_keys.call(e, cursor_offset=self.cursor_offset, line=self.current_line, cut_buffer=self.cut_buffer)
    elif e in key_dispatch[self.config.cut_to_buffer_key]:
        self.cut_to_buffer()
    elif e in key_dispatch[self.config.reimport_key]:
        self.clear_modules_and_reevaluate()
    elif e in key_dispatch[self.config.toggle_file_watch_key]:
        self.toggle_file_watch()
    elif e in key_dispatch[self.config.clear_screen_key]:
        self.request_paint_to_clear_screen = True
    elif e in key_dispatch[self.config.show_source_key]:
        self.show_source()
    elif e in key_dispatch[self.config.help_key]:
        self.pager(self.help_text())
    elif e in key_dispatch[self.config.exit_key]:
        raise SystemExit()
    elif e in ('\n', '\r', '<PADENTER>', '<Ctrl-j>', '<Ctrl-m>'):
        self.on_enter()
    elif e == '<TAB>':
        self.on_tab()
    elif e == '<Shift-TAB>':
        self.on_tab(back=True)
    elif e in key_dispatch[self.config.undo_key]:
        self.prompt_undo()
    elif e in key_dispatch[self.config.redo_key]:
        self.redo()
    elif e in key_dispatch[self.config.save_key]:
        greenlet.greenlet(self.write2file).switch()
    elif e in key_dispatch[self.config.pastebin_key]:
        greenlet.greenlet(self.pastebin).switch()
    elif e in key_dispatch[self.config.copy_clipboard_key]:
        greenlet.greenlet(self.copy2clipboard).switch()
    elif e in key_dispatch[self.config.external_editor_key]:
        self.send_session_to_external_editor()
    elif e in key_dispatch[self.config.edit_config_key]:
        greenlet.greenlet(self.edit_config).switch()
    elif e in key_dispatch[self.config.edit_current_block_key]:
        self.send_current_block_to_external_editor()
    elif e == '<ESC>':
        self.incr_search_mode = SearchMode.NO_SEARCH
    elif e == '<SPACE>':
        self.add_normal_character(' ')
    elif e in CHARACTER_PAIR_MAP.keys():
        if e in ["'", '"']:
            if self.is_closing_quote(e):
                self.insert_char_pair_end(e)
            else:
                self.insert_char_pair_start(e)
        else:
            self.insert_char_pair_start(e)
    elif e in CHARACTER_PAIR_MAP.values():
        self.insert_char_pair_end(e)
    else:
        self.add_normal_character(e)