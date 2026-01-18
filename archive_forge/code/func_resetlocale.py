import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def resetlocale(category=LC_ALL):
    """ Sets the locale for category to the default setting.

        The default setting is determined by calling
        getdefaultlocale(). category defaults to LC_ALL.

    """
    import warnings
    warnings.warn('Use locale.setlocale(locale.LC_ALL, "") instead', DeprecationWarning, stacklevel=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        loc = getdefaultlocale()
    _setlocale(category, _build_localename(loc))