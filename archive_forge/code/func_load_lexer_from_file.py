import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def load_lexer_from_file(filename, lexername='CustomLexer', **options):
    """Load a lexer from a file.

    This method expects a file located relative to the current working
    directory, which contains a Lexer class. By default, it expects the
    Lexer to be name CustomLexer; you can specify your own class name
    as the second argument to this function.

    Users should be very careful with the input, because this method
    is equivalent to running eval on the input file.

    Raises ClassNotFound if there are any problems importing the Lexer.

    .. versionadded:: 2.2
    """
    try:
        custom_namespace = {}
        exec(open(filename, 'rb').read(), custom_namespace)
        if lexername not in custom_namespace:
            raise ClassNotFound('no valid %s class found in %s' % (lexername, filename))
        lexer_class = custom_namespace[lexername]
        return lexer_class(**options)
    except IOError as err:
        raise ClassNotFound('cannot read %s' % filename)
    except ClassNotFound as err:
        raise
    except Exception as err:
        raise ClassNotFound('error when loading custom lexer: %s' % err)