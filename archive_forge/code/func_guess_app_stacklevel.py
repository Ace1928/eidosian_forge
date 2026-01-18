from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
def guess_app_stacklevel(start=1):
    """
    try to guess stacklevel for application warning.
    looks for first frame not part of passlib.
    """
    frame = inspect.currentframe()
    count = -start
    try:
        while frame:
            name = frame.f_globals.get('__name__', '')
            if name.startswith('passlib.tests.') or not name.startswith('passlib.'):
                return max(1, count)
            count += 1
            frame = frame.f_back
        return start
    finally:
        del frame