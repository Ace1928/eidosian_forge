from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
def getorig(self, path, default=None):
    """return original (unpatched) value for path"""
    try:
        value, _ = self._state[path]
    except KeyError:
        value = self._get_path(path)
    return default if value is _UNSET else value