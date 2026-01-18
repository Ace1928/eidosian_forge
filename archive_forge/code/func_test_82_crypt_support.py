from __future__ import with_statement
from binascii import unhexlify
import contextlib
from functools import wraps, partial
import hashlib
import logging; log = logging.getLogger(__name__)
import random
import re
import os
import sys
import tempfile
import threading
import time
from passlib.exc import PasslibHashWarning, PasslibConfigWarning
from passlib.utils.compat import PY3, JYTHON
import warnings
from warnings import warn
from passlib import exc
from passlib.exc import MissingBackendError
import passlib.registry as registry
from passlib.tests.backports import TestCase as _TestCase, skip, skipIf, skipUnless, SkipTest
from passlib.utils import has_rounds_info, has_salt_info, rounds_cost_values, \
from passlib.utils.compat import iteritems, irange, u, unicode, PY2, nullcontext
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
@doesnt_require_backend
def test_82_crypt_support(self):
    """
        test platform-specific crypt() support detection

        NOTE: this is mainly just a sanity check to ensure the runtime
              detection is functioning correctly on some known platforms,
              so that we can feel more confident it'll work right on unknown ones.
        """
    if hasattr(self.handler, 'orig_prefix'):
        raise self.skipTest('not applicable to wrappers')
    using_backend = not self.using_patched_crypt
    name = self.handler.name
    platform = sys.platform
    for pattern, expected in self.platform_crypt_support:
        if re.match(pattern, platform):
            break
    else:
        raise self.skipTest('no data for %r platform (current host support = %r)' % (platform, using_backend))
    if expected is None:
        raise self.skipTest('varied support on %r platform (current host support = %r)' % (platform, using_backend))
    if expected == using_backend:
        pass
    elif expected:
        self.fail('expected %r platform would have native support for %r' % (platform, name))
    else:
        self.fail('did not expect %r platform would have native support for %r' % (platform, name))