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
def indentation_action(self, text):
    self.begin('')
    if text:
        c = text[0]
        if self.indentation_char is None:
            self.indentation_char = c
        elif self.indentation_char != c:
            self.error_at_scanpos('Mixed use of tabs and spaces')
        if text.replace(c, '') != '':
            self.error_at_scanpos('Mixed use of tabs and spaces')
    current_level = self.current_level()
    new_level = len(text)
    if new_level == current_level:
        return
    elif new_level > current_level:
        self.indentation_stack.append(new_level)
        self.produce('INDENT', '')
    else:
        while new_level < self.current_level():
            self.indentation_stack.pop()
            self.produce('DEDENT', '')
        if new_level != self.current_level():
            self.error_at_scanpos('Inconsistent indentation')