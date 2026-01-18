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
def get_scheme_options_with_flag(self, scheme, category):
    """return composite dict of all options set for scheme.
        includes options inherited from 'all' and from default category.
        result can be modified.
        returns (kwds, has_cat_specific_options)
        """
    get_optionmap = self._get_scheme_optionmap
    kwds = get_optionmap('all', None).copy()
    has_cat_options = False
    if category:
        defkwds = kwds.copy()
        kwds.update(get_optionmap('all', category))
    allowed_settings = self.expand_settings(self.get_base_handler(scheme))
    for key in set(kwds).difference(allowed_settings):
        kwds.pop(key)
    if category:
        for key in set(defkwds).difference(allowed_settings):
            defkwds.pop(key)
    other = get_optionmap(scheme, None)
    kwds.update(other)
    if category:
        defkwds.update(other)
        kwds.update(get_optionmap(scheme, category))
        if kwds != defkwds:
            has_cat_options = True
    return (kwds, has_cat_options)