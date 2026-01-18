import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def guess_lexer(_text, **options):
    """Guess a lexer by strong distinctions in the text (eg, shebang)."""
    ft = get_filetype_from_buffer(_text)
    if ft is not None:
        try:
            return get_lexer_by_name(ft, **options)
        except ClassNotFound:
            pass
    best_lexer = [0.0, None]
    for lexer in _iter_lexerclasses():
        rv = lexer.analyse_text(_text)
        if rv == 1.0:
            return lexer(**options)
        if rv > best_lexer[0]:
            best_lexer[:] = (rv, lexer)
    if not best_lexer[0] or best_lexer[1] is None:
        raise ClassNotFound('no lexer matching the text found')
    return best_lexer[1](**options)