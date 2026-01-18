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
def test_43_hash(self):
    """test hash() method"""
    cc = CryptContext(**self.sample_4_dict)
    hash = cc.hash('password')
    self.assertTrue(hash.startswith('$5$rounds=3000$'))
    self.assertTrue(cc.verify('password', hash))
    self.assertFalse(cc.verify('passwordx', hash))
    self.assertRaises(ValueError, cc.copy, sha256_crypt__default_rounds=4000)
    cc = CryptContext(['des_crypt'])
    for secret, kwds in self.nonstring_vectors:
        self.assertRaises(TypeError, cc.hash, secret, **kwds)
    self.assertRaises(KeyError, CryptContext().hash, 'secret')
    self.assertRaises(TypeError, cc.hash, 'secret', category=1)