from __future__ import absolute_import
import cython
import os
import platform
from unicodedata import normalize
from contextlib import contextmanager
from .. import Utils
from ..Plex.Scanners import Scanner
from ..Plex.Errors import UnrecognizedInput
from .Errors import error, warning, hold_errors, release_errors, CompileError
from .Lexicon import any_string_prefix, make_lexicon, IDENT
from .Future import print_function
def exit_async(self):
    assert self.async_enabled > 0
    self.async_enabled -= 1
    if not self.async_enabled:
        del self.keywords['await']
        del self.keywords['async']
        if self.sy in ('async', 'await'):
            self.sy, self.systring = (IDENT, self.context.intern_ustring(self.sy))