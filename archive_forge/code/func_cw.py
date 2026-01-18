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
def cw(self):
    """Return the current word (incomplete word left of cursor)."""
    if self.edit is None:
        return
    pos = self.edit.edit_pos
    text = self.edit.get_edit_text()
    if pos != len(text):
        return
    if not text or (not text[-1].isalnum() and text[-1] not in ('.', '_')):
        return
    for i, c in enumerate(reversed(text)):
        if not c.isalnum() and c not in ('.', '_'):
            break
    else:
        return text
    return text[-i:]