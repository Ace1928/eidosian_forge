from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def get_handler_case(scheme):
    """
    return HandlerCase instance for scheme, used by other tests.

    :param scheme: name of hasher to locate test for (e.g. "bcrypt")

    :raises KeyError:
        if scheme isn't known hasher.

    :raises MissingBackendError:
        if hasher doesn't have any available backends.

    :returns:
        HandlerCase subclass (which derives from TestCase)
    """
    from passlib.registry import get_crypt_handler
    handler = get_crypt_handler(scheme)
    if hasattr(handler, 'backends') and scheme not in _omitted_backend_tests:
        try:
            backend = handler.get_backend()
        except exc.MissingBackendError:
            assert scheme in conditionally_available_hashes
            raise
        name = '%s_%s_test' % (scheme, backend)
    else:
        name = '%s_test' % scheme
    for module in _handler_test_modules:
        modname = 'passlib.tests.' + module
        __import__(modname)
        mod = sys.modules[modname]
        try:
            return getattr(mod, name)
        except AttributeError:
            pass
    raise RuntimeError("can't find test case named %r for %r" % (name, scheme))