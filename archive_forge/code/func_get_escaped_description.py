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
def get_escaped_description(self):
    if self._escaped_description is None:
        esc_desc = self.get_description().encode('ASCII', 'replace').decode('ASCII')
        self._escaped_description = esc_desc.replace('\\', '/')
    return self._escaped_description