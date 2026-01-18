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
def get_error_description(self):
    path = self.filename
    cwd = Utils.decode_filename(os.getcwd() + os.path.sep)
    if path.startswith(cwd):
        return path[len(cwd):]
    return path