import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
def has_os_crypt_support(hasher):
    """
    check if hash is supported by native :func:`crypt.crypt` function.
    if :func:`crypt.crypt` is not present, will always return False.

    :param hasher:
        name or hasher object.

    :returns bool:
        True if hash format is supported by OS, else False.
    """
    return os_crypt_present and has_backend(hasher, OS_CRYPT, safe=True)