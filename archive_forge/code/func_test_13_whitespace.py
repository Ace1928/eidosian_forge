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
def test_13_whitespace(self):
    """whitespace & comment handling"""
    source = to_bytes('\nuser2:pass2\nuser4:pass4\nuser7:pass7\r\n \t \nuser1:pass1\n # legacy users\n#user6:pass6\nuser5:pass5\n\n')
    ht = apache.HtpasswdFile.from_string(source)
    self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user7'])
    ht.set_hash('user4', 'althash4')
    self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user7'])
    ht.set_hash('user6', 'althash6')
    self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user6', 'user7'])
    ht.delete('user7')
    self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user6'])
    target = to_bytes('\nuser2:pass2\nuser4:althash4\n \t \nuser1:pass1\n # legacy users\n#user6:pass6\nuser5:pass5\nuser6:althash6\n')
    self.assertEqual(ht.to_string(), target)