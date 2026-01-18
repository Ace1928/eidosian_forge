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
def default_scheme(self, category=None, resolve=False, unconfigured=False):
    """return name of scheme that :meth:`hash` will use by default.

        :type resolve: bool
        :arg resolve:
            if ``True``, will return a :class:`~passlib.ifc.PasswordHash`
            object instead of the name.

        :type category: str or None
        :param category:
            Optional :ref:`user category <user-categories>`.
            If specified, this will return the catgory-specific default scheme instead.

        :returns:
            name of the default scheme.

        .. seealso:: the :ref:`default <context-default-option>` option for usage example.

        .. versionadded:: 1.6

        .. versionchanged:: 1.7

            This now returns a hasher configured with any CryptContext-specific
            options (custom rounds settings, etc).  Previously this returned
            the base hasher from :mod:`passlib.hash`.
        """
    hasher = self.handler(None, category, unconfigured=unconfigured)
    return hasher if resolve else hasher.name