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
def django_to_passlib_name(self, django_name):
    """
        Convert Django hasher / name to Passlib hasher name.
        """
    return self.django_to_passlib(django_name).name