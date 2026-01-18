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
def parse_mc2(hash, prefix, sep=_UDOLLAR, handler=None):
    """parse hash using 2-part modular crypt format.

    this expects a hash of the format :samp:`{prefix}{salt}[${checksum}]`,
    such as md5_crypt, and parses it into salt / checksum portions.

    :arg hash: the hash to parse (bytes or unicode)
    :arg prefix: the identifying prefix (unicode)
    :param sep: field separator (unicode, defaults to ``$``).
    :param handler: handler class to pass to error constructors.

    :returns:
        a ``(salt, chk | None)`` tuple.
    """
    hash = to_unicode(hash, 'ascii', 'hash')
    assert isinstance(prefix, unicode)
    if not hash.startswith(prefix):
        raise exc.InvalidHashError(handler)
    assert isinstance(sep, unicode)
    parts = hash[len(prefix):].split(sep)
    if len(parts) == 2:
        salt, chk = parts
        return (salt, chk or None)
    elif len(parts) == 1:
        return (parts[0], None)
    else:
        raise exc.MalformedHashError(handler)