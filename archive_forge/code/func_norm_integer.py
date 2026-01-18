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
def norm_integer(handler, value, min=1, max=None, param='value', relaxed=False):
    """
    helper to normalize and validate an integer value (e.g. rounds, salt_size)

    :arg value: value provided to constructor
    :arg default: default value if none provided. if set to ``None``, value is required.
    :arg param: name of parameter (xxx: move to first arg?)
    :param min: minimum value (defaults to 1)
    :param max: maximum value (default ``None`` means no maximum)
    :returns: validated value
    """
    if not isinstance(value, int_types):
        raise exc.ExpectedTypeError(value, 'integer', param)
    if value < min:
        msg = '%s: %s (%d) is too low, must be at least %d' % (handler.name, param, value, min)
        if relaxed:
            warn(msg, exc.PasslibHashWarning)
            value = min
        else:
            raise ValueError(msg)
    if max and value > max:
        msg = '%s: %s (%d) is too large, cannot be more than %d' % (handler.name, param, value, max)
        if relaxed:
            warn(msg, exc.PasslibHashWarning)
            value = max
        else:
            raise ValueError(msg)
    return value