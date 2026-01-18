from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
def test_20_norm_salt(self):
    """test GenericHandler + HasSalt mixin"""

    class d1(uh.HasSalt, uh.GenericHandler):
        name = 'd1'
        setting_kwds = ('salt',)
        min_salt_size = 2
        max_salt_size = 4
        default_salt_size = 3
        salt_chars = 'ab'

    def norm_salt(**k):
        return d1(**k).salt

    def gen_salt(sz, **k):
        return d1.using(salt_size=sz, **k)(use_defaults=True).salt
    salts2 = _makelang('ab', 2)
    salts3 = _makelang('ab', 3)
    salts4 = _makelang('ab', 4)
    self.assertRaises(TypeError, norm_salt)
    self.assertRaises(TypeError, norm_salt, salt=None)
    self.assertIn(norm_salt(use_defaults=True), salts3)
    with warnings.catch_warnings(record=True) as wlog:
        self.assertRaises(ValueError, norm_salt, salt='')
        self.assertRaises(ValueError, norm_salt, salt='a')
        self.consumeWarningList(wlog)
        self.assertEqual(norm_salt(salt='ab'), 'ab')
        self.assertEqual(norm_salt(salt='aba'), 'aba')
        self.assertEqual(norm_salt(salt='abba'), 'abba')
        self.consumeWarningList(wlog)
        self.assertRaises(ValueError, norm_salt, salt='aaaabb')
        self.consumeWarningList(wlog)
    with warnings.catch_warnings(record=True) as wlog:
        self.assertRaises(ValueError, gen_salt, 0)
        self.assertRaises(ValueError, gen_salt, 1)
        self.consumeWarningList(wlog)
        self.assertIn(gen_salt(2), salts2)
        self.assertIn(gen_salt(3), salts3)
        self.assertIn(gen_salt(4), salts4)
        self.consumeWarningList(wlog)
        self.assertRaises(ValueError, gen_salt, 5)
        self.consumeWarningList(wlog)
        self.assertIn(gen_salt(5, relaxed=True), salts4)
        self.consumeWarningList(wlog, ['salt_size.*above max_salt_size'])
    del d1.max_salt_size
    with self.assertWarningList([]):
        self.assertEqual(len(gen_salt(None)), 3)
        self.assertEqual(len(gen_salt(5)), 5)