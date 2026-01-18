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
def set_edit_pos(self, pos):
    super().set_edit_pos(pos)
    self._emit('edit-pos-changed', self.edit_pos)