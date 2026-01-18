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
def parse_int(source, base=10, default=None, param='value', handler=None):
    """
    helper to parse an integer config field

    :arg source: unicode source string
    :param base: numeric base
    :param default: optional default if source is empty
    :param param: name of variable, for error msgs
    :param handler: handler class, for error msgs
    """
    if source.startswith(_UZERO) and source != _UZERO:
        raise exc.MalformedHashError(handler, 'zero-padded %s field' % param)
    elif source:
        return int(source, base)
    elif default is None:
        raise exc.MalformedHashError(handler, 'empty %s field' % param)
    else:
        return default