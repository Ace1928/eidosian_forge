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
def identify_record(self, hash, category, required=True):
    """internal helper to identify appropriate custom handler for hash"""
    if not isinstance(hash, unicode_or_bytes_types):
        raise ExpectedStringError(hash, 'hash')
    for record in self._get_record_list(category):
        if record.identify(hash):
            return record
    if not required:
        return None
    elif not self.schemes:
        raise KeyError('no crypt algorithms supported')
    else:
        raise exc.UnknownHashError('hash could not be identified')