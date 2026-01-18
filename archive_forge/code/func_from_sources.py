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
@classmethod
def from_sources(cls, sources, _warn=True):
    """create a CryptPolicy instance by merging multiple sources.

        each source is interpreted as by :meth:`from_source`,
        and the results are merged together.

        .. deprecated:: 1.6
            Instead of using this method to merge multiple policies together,
            a :class:`CryptContext` instance should be created, and then
            the multiple sources merged together via :meth:`CryptContext.load`.
        """
    if _warn:
        warn(_preamble + 'Instead of ``CryptPolicy.from_sources()``, use the various CryptContext constructors  followed by ``context.update()``.', DeprecationWarning, stacklevel=2)
    if len(sources) == 0:
        raise ValueError('no sources specified')
    if len(sources) == 1:
        return cls.from_source(sources[0], _warn=False)
    kwds = {}
    for source in sources:
        kwds.update(cls.from_source(source, _warn=False)._context.to_dict(resolve=True))
    return cls(_internal_context=CryptContext(**kwds))