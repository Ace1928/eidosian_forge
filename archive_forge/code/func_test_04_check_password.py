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
def test_04_check_password(self):
    """test check_password()"""
    ht = apache.HtdigestFile.from_string(self.sample_01)
    self.assertRaises(TypeError, ht.check_password, 1, 'realm', 'pass5')
    self.assertRaises(TypeError, ht.check_password, 'user', 1, 'pass5')
    self.assertIs(ht.check_password('user5', 'realm', 'pass5'), None)
    for i in irange(1, 5):
        i = str(i)
        self.assertTrue(ht.check_password('user' + i, 'realm', 'pass' + i))
        self.assertIs(ht.check_password('user' + i, 'realm', 'pass5'), False)
    self.assertRaises(TypeError, ht.check_password, 'user5', 'pass5')
    ht.default_realm = 'realm'
    self.assertTrue(ht.check_password('user1', 'pass1'))
    self.assertIs(ht.check_password('user5', 'pass5'), None)
    with self.assertWarningList(['verify\\(\\) is deprecated'] * 2):
        self.assertTrue(ht.verify('user1', 'realm', 'pass1'))
        self.assertFalse(ht.verify('user1', 'realm', 'pass2'))
    self.assertRaises(ValueError, ht.check_password, 'user:', 'realm', 'pass')