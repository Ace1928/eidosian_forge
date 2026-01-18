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
def test_50_rounds_limits(self):
    """test rounds limits"""
    cc = CryptContext(schemes=['sha256_crypt'], sha256_crypt__min_rounds=2000, sha256_crypt__max_rounds=3000, sha256_crypt__default_rounds=2500)
    STUB = '...........................................'
    custom_handler = cc._get_record('sha256_crypt', None)
    self.assertEqual(custom_handler.min_desired_rounds, 2000)
    self.assertEqual(custom_handler.max_desired_rounds, 3000)
    self.assertEqual(custom_handler.default_rounds, 2500)
    with self.assertWarningList([PasslibHashWarning] * 2):
        c2 = cc.copy(sha256_crypt__min_rounds=500, sha256_crypt__max_rounds=None, sha256_crypt__default_rounds=500)
    self.assertEqual(c2.genconfig(salt='nacl'), '$5$rounds=1000$nacl$' + STUB)
    with self.assertWarningList([]):
        self.assertEqual(cc.genconfig(rounds=1999, salt='nacl'), '$5$rounds=1999$nacl$' + STUB)
    self.assertEqual(cc.genconfig(rounds=2000, salt='nacl'), '$5$rounds=2000$nacl$' + STUB)
    self.assertEqual(cc.genconfig(rounds=2001, salt='nacl'), '$5$rounds=2001$nacl$' + STUB)
    with self.assertWarningList([PasslibHashWarning] * 2):
        c2 = cc.copy(sha256_crypt__max_rounds=int(1000000000.0) + 500, sha256_crypt__min_rounds=None, sha256_crypt__default_rounds=int(1000000000.0) + 500)
    self.assertEqual(c2.genconfig(salt='nacl'), '$5$rounds=999999999$nacl$' + STUB)
    with self.assertWarningList([]):
        self.assertEqual(cc.genconfig(rounds=3001, salt='nacl'), '$5$rounds=3001$nacl$' + STUB)
    self.assertEqual(cc.genconfig(rounds=3000, salt='nacl'), '$5$rounds=3000$nacl$' + STUB)
    self.assertEqual(cc.genconfig(rounds=2999, salt='nacl'), '$5$rounds=2999$nacl$' + STUB)
    self.assertEqual(cc.genconfig(salt='nacl'), '$5$rounds=2500$nacl$' + STUB)
    df = hash.sha256_crypt.default_rounds
    c2 = cc.copy(sha256_crypt__default_rounds=None, sha256_crypt__max_rounds=df << 1)
    self.assertEqual(c2.genconfig(salt='nacl'), '$5$rounds=%d$nacl$%s' % (df, STUB))
    c2 = cc.copy(sha256_crypt__default_rounds=None, sha256_crypt__max_rounds=3000)
    self.assertEqual(c2.genconfig(salt='nacl'), '$5$rounds=3000$nacl$' + STUB)
    self.assertRaises(ValueError, cc.copy, sha256_crypt__default_rounds=1999)
    cc.copy(sha256_crypt__default_rounds=2000)
    cc.copy(sha256_crypt__default_rounds=3000)
    self.assertRaises(ValueError, cc.copy, sha256_crypt__default_rounds=3001)
    c2 = CryptContext(schemes=['sha256_crypt'])
    self.assertRaises(ValueError, c2.copy, sha256_crypt__min_rounds=2000, sha256_crypt__max_rounds=1999)
    self.assertRaises(ValueError, CryptContext, sha256_crypt__min_rounds='x')
    self.assertRaises(ValueError, CryptContext, sha256_crypt__max_rounds='x')
    self.assertRaises(ValueError, CryptContext, all__vary_rounds='x')
    self.assertRaises(ValueError, CryptContext, sha256_crypt__default_rounds='x')
    bad = datetime.datetime.now()
    self.assertRaises(TypeError, CryptContext, 'sha256_crypt', sha256_crypt__min_rounds=bad)
    self.assertRaises(TypeError, CryptContext, 'sha256_crypt', sha256_crypt__max_rounds=bad)
    self.assertRaises(TypeError, CryptContext, 'sha256_crypt', all__vary_rounds=bad)
    self.assertRaises(TypeError, CryptContext, 'sha256_crypt', sha256_crypt__default_rounds=bad)