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
def iter_handlers(self):
    """return iterator over handlers defined in policy.

        .. deprecated:: 1.6
            applications should use ``context.schemes(resolve=True))`` instead.
            see :meth:`CryptContext.schemes`.
        """
    if self._stub_policy:
        warn(_preamble + 'Instead of ``context.policy.iter_handlers()``, use ``context.schemes(resolve=True)``.', DeprecationWarning, stacklevel=2)
    else:
        warn(_preamble + 'Instead of ``CryptPolicy().iter_handlers()``, create a CryptContext instance and use ``context.schemes(resolve=True)``.', DeprecationWarning, stacklevel=2)
    return self._context.schemes(resolve=True, unconfigured=True)