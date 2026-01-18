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
@contextmanager
@cython.locals(scanner=Scanner)
def tentatively_scan(scanner):
    errors = hold_errors()
    try:
        put_back_on_failure = scanner.put_back_on_failure
        scanner.put_back_on_failure = []
        initial_state = (scanner.sy, scanner.systring, scanner.position())
        try:
            yield errors
        except CompileError as e:
            pass
        finally:
            if errors:
                if scanner.put_back_on_failure:
                    for put_back in reversed(scanner.put_back_on_failure[:-1]):
                        scanner.put_back(*put_back)
                    scanner.put_back(*initial_state)
            elif put_back_on_failure is not None:
                put_back_on_failure.extend(scanner.put_back_on_failure)
            scanner.put_back_on_failure = put_back_on_failure
    finally:
        release_errors(ignore=True)