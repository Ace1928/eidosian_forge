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
def get_user_category(self, user):
    """
        Helper for hashing passwords per-user --
        figure out the CryptContext category for specified Django user object.
        .. note::
            This may be overridden via PASSLIB_GET_CATEGORY django setting
        """
    if user.is_superuser:
        return 'superuser'
    elif user.is_staff:
        return 'staff'
    else:
        return None