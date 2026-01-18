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
def schemes(self, resolve=False, category=None, unconfigured=False):
    """return schemes loaded into this CryptContext instance.

        :type resolve: bool
        :arg resolve:
            if ``True``, will return a tuple of :class:`~passlib.ifc.PasswordHash`
            objects instead of their names.

        :returns:
            returns tuple of the schemes configured for this context
            via the *schemes* option.

        .. versionadded:: 1.6
            This was previously available as ``CryptContext().policy.schemes()``

        .. seealso:: the :ref:`schemes <context-schemes-option>` option for usage example.
        """
    schemes = self._config.schemes
    if resolve:
        return tuple((self.handler(scheme, category, unconfigured=unconfigured) for scheme in schemes))
    else:
        return schemes