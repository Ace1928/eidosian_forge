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
def test_01_delete_autosave(self):
    path = self.mktemp()
    set_file(path, self.sample_01)
    ht = apache.HtdigestFile(path)
    self.assertTrue(ht.delete('user1', 'realm'))
    self.assertFalse(ht.delete('user3', 'realm5'))
    self.assertFalse(ht.delete('user5', 'realm'))
    self.assertEqual(get_file(path), self.sample_01)
    ht.autosave = True
    self.assertTrue(ht.delete('user2', 'realm'))
    self.assertEqual(get_file(path), self.sample_02)