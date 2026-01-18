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
def test_03_users(self):
    """test users()"""
    ht = apache.HtdigestFile.from_string(self.sample_01)
    ht.set_password('user5', 'realm', 'pass5')
    ht.delete('user3', 'realm')
    ht.set_password('user3', 'realm', 'pass3')
    self.assertEqual(sorted(ht.users('realm')), ['user1', 'user2', 'user3', 'user4', 'user5'])
    self.assertRaises(TypeError, ht.users, 1)