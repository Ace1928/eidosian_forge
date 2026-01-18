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
def test_12_ident(self):
    h = uh.PrefixWrapper('h2', 'ldap_md5', '{XXX}')
    self.assertEqual(h.ident, u('{XXX}{MD5}'))
    self.assertIs(h.ident_values, None)
    h = uh.PrefixWrapper('h2', 'des_crypt', '{XXX}')
    self.assertIs(h.ident, None)
    self.assertIs(h.ident_values, None)
    h = uh.PrefixWrapper('h1', 'ldap_md5', '{XXX}', '{MD5}')
    self.assertIs(h.ident, None)
    self.assertIs(h.ident_values, None)
    h = uh.PrefixWrapper('h3', 'ldap_md5', '{XXX}', ident='{X')
    self.assertEqual(h.ident, u('{X'))
    self.assertIs(h.ident_values, None)
    h = uh.PrefixWrapper('h3', 'ldap_md5', '{XXX}', ident='{XXX}A')
    self.assertRaises(ValueError, uh.PrefixWrapper, 'h3', 'ldap_md5', '{XXX}', ident='{XY')
    self.assertRaises(ValueError, uh.PrefixWrapper, 'h3', 'ldap_md5', '{XXX}', ident='{XXXX')
    h = uh.PrefixWrapper('h4', 'phpass', '{XXX}')
    self.assertIs(h.ident, None)
    self.assertEqual(h.ident_values, (u('{XXX}$P$'), u('{XXX}$H$')))
    h = uh.PrefixWrapper('h5', 'des_crypt', '{XXX}', ident=True)
    self.assertEqual(h.ident, u('{XXX}'))
    self.assertIs(h.ident_values, None)
    self.assertRaises(ValueError, uh.PrefixWrapper, 'h6', 'des_crypt', ident=True)
    with self.assertWarningList('orig_prefix.*may not work correctly'):
        h = uh.PrefixWrapper('h7', 'phpass', orig_prefix='$', prefix='?')
    self.assertEqual(h.ident_values, None)
    self.assertEqual(h.ident, None)