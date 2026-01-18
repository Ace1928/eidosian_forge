import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def get_lexer_by_name(_alias, **options):
    """Get a lexer by an alias.

    Raises ClassNotFound if not found.
    """
    if not _alias:
        raise ClassNotFound('no lexer for alias %r found' % _alias)
    for module_name, name, aliases, _, _ in itervalues(LEXERS):
        if _alias.lower() in aliases:
            if name not in _lexer_cache:
                _load_lexers(module_name)
            return _lexer_cache[name](**options)
    for cls in find_plugin_lexers():
        if _alias.lower() in cls.aliases:
            return cls(**options)
    raise ClassNotFound('no lexer for alias %r found' % _alias)