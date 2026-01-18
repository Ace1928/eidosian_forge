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
@classmethod
def peek_unpatched_func(cls, value):
    return value._patched_original_value