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
def test_00_constructor_autoload(self):
    """test constructor autoload"""
    path = self.mktemp()
    set_file(path, self.sample_01)
    ht = apache.HtdigestFile(path)
    self.assertEqual(ht.to_string(), self.sample_01)
    ht = apache.HtdigestFile(path, new=True)
    self.assertEqual(ht.to_string(), b'')
    os.remove(path)
    self.assertRaises(IOError, apache.HtdigestFile, path)