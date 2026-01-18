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
def test_02_set_password(self):
    """test update()"""
    ht = apache.HtdigestFile.from_string(self.sample_01)
    self.assertTrue(ht.set_password('user2', 'realm', 'pass2x'))
    self.assertFalse(ht.set_password('user5', 'realm', 'pass5'))
    self.assertEqual(ht.to_string(), self.sample_03)
    self.assertRaises(TypeError, ht.set_password, 'user2', 'pass3')
    ht.default_realm = 'realm2'
    ht.set_password('user2', 'pass3')
    ht.check_password('user2', 'realm2', 'pass3')
    self.assertRaises(ValueError, ht.set_password, 'user:', 'realm', 'pass')
    self.assertRaises(ValueError, ht.set_password, 'u' * 256, 'realm', 'pass')
    self.assertRaises(ValueError, ht.set_password, 'user', 'realm:', 'pass')
    self.assertRaises(ValueError, ht.set_password, 'user', 'r' * 256, 'pass')
    with self.assertWarningList('update\\(\\) is deprecated'):
        ht.update('user2', 'realm2', 'test')
    self.assertTrue(ht.check_password('user2', 'test'))