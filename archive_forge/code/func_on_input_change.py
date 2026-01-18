import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def on_input_change(self, edit, text):
    tokens = self.tokenize(text, False)
    edit.set_edit_markup(list(format_tokens(tokens)))
    if not self._completion_update_suppressed:
        self.main_loop.set_alarm_in(0, lambda *args: self._populate_completion())