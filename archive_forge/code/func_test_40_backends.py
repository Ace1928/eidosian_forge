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
def test_40_backends(self):
    """test GenericHandler + HasManyBackends mixin"""

    class d1(uh.HasManyBackends, uh.GenericHandler):
        name = 'd1'
        setting_kwds = ()
        backends = ('a', 'b')
        _enable_a = False
        _enable_b = False

        @classmethod
        def _load_backend_a(cls):
            if cls._enable_a:
                cls._set_calc_checksum_backend(cls._calc_checksum_a)
                return True
            else:
                return False

        @classmethod
        def _load_backend_b(cls):
            if cls._enable_b:
                cls._set_calc_checksum_backend(cls._calc_checksum_b)
                return True
            else:
                return False

        def _calc_checksum_a(self, secret):
            return 'a'

        def _calc_checksum_b(self, secret):
            return 'b'
    self.assertRaises(MissingBackendError, d1.get_backend)
    self.assertRaises(MissingBackendError, d1.set_backend)
    self.assertRaises(MissingBackendError, d1.set_backend, 'any')
    self.assertRaises(MissingBackendError, d1.set_backend, 'default')
    self.assertFalse(d1.has_backend())
    d1._enable_b = True
    obj = d1()
    self.assertEqual(obj._calc_checksum('s'), 'b')
    d1.set_backend('b')
    d1.set_backend('any')
    self.assertEqual(obj._calc_checksum('s'), 'b')
    self.assertRaises(MissingBackendError, d1.set_backend, 'a')
    self.assertTrue(d1.has_backend('b'))
    self.assertFalse(d1.has_backend('a'))
    d1._enable_a = True
    self.assertTrue(d1.has_backend())
    d1.set_backend('a')
    self.assertEqual(obj._calc_checksum('s'), 'a')
    self.assertRaises(ValueError, d1.set_backend, 'c')
    self.assertRaises(ValueError, d1.has_backend, 'c')
    d1.set_backend('b')

    class d2(d1):
        _has_backend_a = True
    self.assertRaises(AssertionError, d2.has_backend, 'a')