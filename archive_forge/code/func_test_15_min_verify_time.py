from __future__ import with_statement
from logging import getLogger
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, CryptPolicy, LazyCryptContext
from passlib.utils import to_bytes, to_unicode
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase, set_file
from passlib.registry import (register_crypt_handler_path,
def test_15_min_verify_time(self):
    """test get_min_verify_time() method"""
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    pa = CryptPolicy()
    self.assertEqual(pa.get_min_verify_time(), 0)
    self.assertEqual(pa.get_min_verify_time('admin'), 0)
    pb = pa.replace(min_verify_time=0.1)
    self.assertEqual(pb.get_min_verify_time(), 0)
    self.assertEqual(pb.get_min_verify_time('admin'), 0)