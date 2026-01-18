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
def passlib_to_django_name(self, passlib_name):
    """
        Convert passlib hasher / name to Django hasher name.
        """
    return self.passlib_to_django(passlib_name).algorithm