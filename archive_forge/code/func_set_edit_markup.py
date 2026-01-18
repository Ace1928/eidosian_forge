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
def set_edit_markup(self, markup):
    """Call this when markup changes but the underlying text does not.

        You should arrange for this to be called from the 'change' signal.
        """
    if markup:
        self._bpy_text, self._bpy_attr = urwid.decompose_tagmarkup(markup)
    else:
        self._bpy_text, self._bpy_attr = ('', [])
    self._invalidate()