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
@classmethod
def create_backend_case(cls, backend):
    handler = cls.handler
    name = handler.name
    assert hasattr(handler, 'backends'), 'handler must support uh.HasManyBackends protocol'
    assert backend in handler.backends, 'unknown backend: %r' % (backend,)
    bases = (cls,)
    if backend == 'os_crypt':
        bases += (OsCryptMixin,)
    subcls = type('%s_%s_test' % (name, backend), bases, dict(descriptionPrefix='%s (%s backend)' % (name, backend), backend=backend, _skip_backend_reason=cls._get_skip_backend_reason(backend), __module__=cls.__module__))
    return subcls