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
def test_05_load(self):
    """test load()"""
    path = self.mktemp()
    set_file(path, '')
    backdate_file_mtime(path, 5)
    ha = apache.HtdigestFile(path)
    self.assertEqual(ha.to_string(), b'')
    ha.set_password('user1', 'realm', 'pass1')
    ha.load_if_changed()
    self.assertEqual(ha.to_string(), b'user1:realm:2a6cf53e7d8f8cf39d946dc880b14128\n')
    set_file(path, self.sample_01)
    ha.load_if_changed()
    self.assertEqual(ha.to_string(), self.sample_01)
    ha.set_password('user5', 'realm', 'pass5')
    ha.load()
    self.assertEqual(ha.to_string(), self.sample_01)
    hb = apache.HtdigestFile()
    self.assertRaises(RuntimeError, hb.load)
    self.assertRaises(RuntimeError, hb.load_if_changed)
    hc = apache.HtdigestFile()
    hc.load(path)
    self.assertEqual(hc.to_string(), self.sample_01)
    ensure_mtime_changed(path)
    set_file(path, '')
    with self.assertWarningList('load\\(force=False\\) is deprecated'):
        ha.load(force=False)
    self.assertEqual(ha.to_string(), b'')