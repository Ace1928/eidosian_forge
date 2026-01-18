import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def find_lexer_class_for_filename(_fn, code=None):
    """Get a lexer for a filename.

    If multiple lexers match the filename pattern, use ``analyse_text()`` to
    figure out which one is more appropriate.

    Returns None if not found.
    """
    matches = []
    fn = basename(_fn)
    for modname, name, _, filenames, _ in itervalues(LEXERS):
        for filename in filenames:
            if _fn_matches(fn, filename):
                if name not in _lexer_cache:
                    _load_lexers(modname)
                matches.append((_lexer_cache[name], filename))
    for cls in find_plugin_lexers():
        for filename in cls.filenames:
            if _fn_matches(fn, filename):
                matches.append((cls, filename))
    if sys.version_info > (3,) and isinstance(code, bytes):
        code = guess_decode(code)

    def get_rating(info):
        cls, filename = info
        bonus = '*' not in filename and 0.5 or 0
        if code:
            return (cls.analyse_text(code) + bonus, cls.__name__)
        return (cls.priority + bonus, cls.__name__)
    if matches:
        matches.sort(key=get_rating)
        return matches[-1][0]