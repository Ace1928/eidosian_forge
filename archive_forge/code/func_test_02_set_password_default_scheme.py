from __future__ import with_statement
from logging import getLogger
import os
import subprocess
from passlib import apache, registry
from passlib.exc import MissingBackendError
from passlib.utils.compat import irange
from passlib.tests.backports import unittest
from passlib.tests.utils import TestCase, get_file, set_file, ensure_mtime_changed
from passlib.utils.compat import u
from passlib.utils import to_bytes
from passlib.utils.handlers import to_unicode_for_identify
def test_02_set_password_default_scheme(self):
    """test set_password() -- default_scheme"""

    def check(scheme):
        ht = apache.HtpasswdFile(default_scheme=scheme)
        ht.set_password('user1', 'pass1')
        return ht.context.identify(ht.get_hash('user1'))
    self.assertEqual(check('sha256_crypt'), 'sha256_crypt')
    self.assertEqual(check('des_crypt'), 'des_crypt')
    self.assertRaises(KeyError, check, 'xxx')
    self.assertEqual(check('portable'), apache.htpasswd_defaults['portable'])
    self.assertEqual(check('portable_apache_22'), apache.htpasswd_defaults['portable_apache_22'])
    self.assertEqual(check('host_apache_22'), apache.htpasswd_defaults['host_apache_22'])
    self.assertEqual(check(None), apache.htpasswd_defaults['portable_apache_22'])