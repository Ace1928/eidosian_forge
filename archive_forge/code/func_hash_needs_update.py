from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
@deprecated_method(deprecated='1.6', removed='2.0', replacement='CryptContext.needs_update()')
def hash_needs_update(self, hash, scheme=None, category=None):
    """Legacy alias for :meth:`needs_update`.

        .. deprecated:: 1.6
            This method was renamed to :meth:`!needs_update` in version 1.6.
            This alias will be removed in version 2.0, and should only
            be used for compatibility with Passlib 1.3 - 1.5.
        """
    return self.needs_update(hash, scheme, category)