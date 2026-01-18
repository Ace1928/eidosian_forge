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
def test_11_wrapped_methods(self):
    d1 = uh.PrefixWrapper('d1', 'ldap_md5', '{XXX}', '{MD5}')
    dph = '{XXX}X03MO1qnZdYdgyfeuILPmQ=='
    lph = '{MD5}X03MO1qnZdYdgyfeuILPmQ=='
    self.assertEqual(d1.genconfig(), '{XXX}1B2M2Y8AsgTpgAmY7PhCfg==')
    self.assertRaises(TypeError, d1.genhash, 'password', None)
    self.assertEqual(d1.genhash('password', dph), dph)
    self.assertRaises(ValueError, d1.genhash, 'password', lph)
    self.assertEqual(d1.hash('password'), dph)
    self.assertTrue(d1.identify(dph))
    self.assertFalse(d1.identify(lph))
    self.assertRaises(ValueError, d1.verify, 'password', lph)
    self.assertTrue(d1.verify('password', dph))