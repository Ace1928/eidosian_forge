from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def test_90_variant(self):
    """test variant keyword"""
    handler = self.handler
    kwds = dict(salt=b'a', rounds=1)
    handler(variant=1, **kwds)
    handler(variant=u('1'), **kwds)
    handler(variant=b'1', **kwds)
    handler(variant=u('sha256'), **kwds)
    handler(variant=b'sha256', **kwds)
    self.assertRaises(TypeError, handler, variant=None, **kwds)
    self.assertRaises(TypeError, handler, variant=complex(1, 1), **kwds)
    self.assertRaises(ValueError, handler, variant='9', **kwds)
    self.assertRaises(ValueError, handler, variant=9, **kwds)