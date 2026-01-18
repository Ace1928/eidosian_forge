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
def is_deprecated_with_flag(self, scheme, category):
    """is scheme deprecated under particular category?"""
    depmap = self.get_context_optionmap('deprecated')

    def test(cat):
        source = depmap.get(cat, depmap.get(None))
        if source is None:
            return None
        elif 'auto' in source:
            return scheme != self.default_scheme(cat)
        else:
            return scheme in source
    value = test(None) or False
    if category:
        alt = test(category)
        if alt is not None and value != alt:
            return (alt, True)
    return (value, False)