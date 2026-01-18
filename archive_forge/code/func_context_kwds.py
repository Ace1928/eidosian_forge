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
@property
def context_kwds(self):
    """
        return :class:`!set` containing union of all :ref:`contextual keywords <context-keywords>`
        supported by the handlers in this context.

        .. versionadded:: 1.6.6
        """
    return self._config.context_kwds