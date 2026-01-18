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
def iter_config(self, resolve=False):
    """regenerate original config.

        this is an iterator which yields ``(cat,scheme,option),value`` items,
        in the order they generally appear inside an INI file.
        if interpreted as a dictionary, it should match the original
        keywords passed to the CryptContext (aside from any canonization).

        it's mainly used as the internal backend for most of the public
        serialization methods.
        """
    scheme_options = self._scheme_options
    context_options = self._context_options
    scheme_keys = sorted(scheme_options)
    context_keys = sorted(context_options)
    if 'schemes' in context_keys:
        context_keys.remove('schemes')
    value = self.handlers if resolve else self.schemes
    if value:
        yield ((None, None, 'schemes'), list(value))
    for cat in (None,) + self.categories:
        for key in context_keys:
            try:
                value = context_options[key][cat]
            except KeyError:
                pass
            else:
                if isinstance(value, list):
                    value = list(value)
                yield ((cat, None, key), value)
        for scheme in scheme_keys:
            try:
                kwds = scheme_options[scheme][cat]
            except KeyError:
                pass
            else:
                for key in sorted(kwds):
                    yield ((cat, scheme, key), kwds[key])