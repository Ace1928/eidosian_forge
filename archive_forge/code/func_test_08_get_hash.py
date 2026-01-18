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
def test_08_get_hash(self):
    """test get_hash()"""
    ht = apache.HtdigestFile.from_string(self.sample_01)
    self.assertEqual(ht.get_hash('user3', 'realm'), 'a500bb8c02f6a9170ae46af10c898744')
    self.assertEqual(ht.get_hash('user4', 'realm'), 'ab7b5d5f28ccc7666315f508c7358519')
    self.assertEqual(ht.get_hash('user5', 'realm'), None)
    with self.assertWarningList('find\\(\\) is deprecated'):
        self.assertEqual(ht.find('user4', 'realm'), 'ab7b5d5f28ccc7666315f508c7358519')