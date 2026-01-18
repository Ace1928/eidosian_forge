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
def handler_is_deprecated(self, name, category=None):
    """check if handler has been deprecated by policy.

        .. deprecated:: 1.6
            this method has no direct replacement in the 1.6 api, as there
            is not a clearly defined use-case. however, examining the output of
            :meth:`CryptContext.to_dict` should serve as the closest alternative.
        """
    if self._stub_policy:
        warn(_preamble + '``context.policy.handler_is_deprecated()`` will no longer be available.', DeprecationWarning, stacklevel=2)
    else:
        warn(_preamble + '``CryptPolicy().handler_is_deprecated()`` will no longer be available.', DeprecationWarning, stacklevel=2)
    if hasattr(name, 'name'):
        name = name.name
    return self._context.handler(name, category).deprecated