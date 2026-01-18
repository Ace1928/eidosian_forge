import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
@memoize_single_value
def get_supported_os_crypt_schemes():
    """
    return tuple of schemes which :func:`crypt.crypt` natively supports.
    """
    if not os_crypt_present:
        return ()
    cache = tuple((name for name in os_crypt_schemes if get_crypt_handler(name).has_backend(OS_CRYPT)))
    if not cache:
        import platform
        warn("crypt.crypt() function is present, but doesn't support any formats known to passlib! (system=%r release=%r)" % (platform.system(), platform.release()), exc.PasslibRuntimeWarning)
    return cache