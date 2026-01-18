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
@property
def ident_values(self):
    value = self._ident_values
    if value is False:
        value = None
        if not self.orig_prefix:
            wrapped = self.wrapped
            idents = getattr(wrapped, 'ident_values', None)
            if idents:
                value = tuple((self._wrap_hash(ident) for ident in idents))
        self._ident_values = value
    return value