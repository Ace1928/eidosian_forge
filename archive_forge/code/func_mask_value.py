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
def mask_value(value, show=4, pct=0.125, char=u'*'):
    """
    helper to mask contents of sensitive field.

    :param value:
        raw value (str, bytes, etc)

    :param show:
        max # of characters to remain visible

    :param pct:
        don't show more than this % of input.

    :param char:
        character to use for masking

    :rtype: str | None
    """
    if value is None:
        return None
    if not isinstance(value, unicode):
        if isinstance(value, bytes):
            from passlib.utils.binary import ab64_encode
            value = ab64_encode(value).decode('ascii')
        else:
            value = unicode(value)
    size = len(value)
    show = min(show, int(size * pct))
    return value[:show] + char * (size - show)