import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def getlocale(category=LC_CTYPE):
    """ Returns the current setting for the given locale category as
        tuple (language code, encoding).

        category may be one of the LC_* value except LC_ALL. It
        defaults to LC_CTYPE.

        Except for the code 'C', the language code corresponds to RFC
        1766.  code and encoding can be None in case the values cannot
        be determined.

    """
    localename = _setlocale(category)
    if category == LC_ALL and ';' in localename:
        raise TypeError('category LC_ALL is not supported')
    return _parse_localename(localename)