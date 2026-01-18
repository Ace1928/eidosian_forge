import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def get_lexer_for_mimetype(_mime, **options):
    """Get a lexer for a mimetype.

    Raises ClassNotFound if not found.
    """
    for modname, name, _, _, mimetypes in itervalues(LEXERS):
        if _mime in mimetypes:
            if name not in _lexer_cache:
                _load_lexers(modname)
            return _lexer_cache[name](**options)
    for cls in find_plugin_lexers():
        if _mime in cls.mimetypes:
            return cls(**options)
    raise ClassNotFound('no lexer for mimetype %r found' % _mime)