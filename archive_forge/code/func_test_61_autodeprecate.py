from __future__ import with_statement
from passlib.utils.compat import PY3
import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, LazyCryptContext
from passlib.exc import PasslibConfigWarning, PasslibHashWarning
from passlib.utils import tick, to_unicode
from passlib.utils.compat import irange, u, unicode, str_to_uascii, PY2, PY26
import passlib.utils.handlers as uh
from passlib.tests.utils import (TestCase, set_file, TICK_RESOLUTION,
from passlib.registry import (register_crypt_handler_path,
import hashlib, time
def test_61_autodeprecate(self):
    """test deprecated='auto' is handled correctly"""

    def getstate(ctx, category=None):
        return [ctx.handler(scheme, category).deprecated for scheme in ctx.schemes()]
    ctx = CryptContext('sha256_crypt,md5_crypt,des_crypt', deprecated='auto')
    self.assertEqual(getstate(ctx, None), [False, True, True])
    self.assertEqual(getstate(ctx, 'admin'), [False, True, True])
    ctx.update(default='md5_crypt')
    self.assertEqual(getstate(ctx, None), [True, False, True])
    self.assertEqual(getstate(ctx, 'admin'), [True, False, True])
    ctx.update(admin__context__default='des_crypt')
    self.assertEqual(getstate(ctx, None), [True, False, True])
    self.assertEqual(getstate(ctx, 'admin'), [True, True, False])
    ctx = CryptContext(['sha256_crypt'], deprecated='auto')
    self.assertEqual(getstate(ctx, None), [False])
    self.assertEqual(getstate(ctx, 'admin'), [False])
    self.assertRaises(ValueError, CryptContext, 'sha256_crypt,md5_crypt', deprecated='auto,md5_crypt')
    self.assertRaises(ValueError, CryptContext, 'sha256_crypt,md5_crypt', deprecated='md5_crypt,auto')