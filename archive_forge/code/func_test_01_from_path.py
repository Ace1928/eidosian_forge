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
def test_01_from_path(self):
    """test CryptPolicy.from_path() constructor with encodings"""
    path = self.mktemp()
    set_file(path, self.sample_config_1s)
    policy = CryptPolicy.from_path(path)
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    set_file(path, self.sample_config_1s.replace('\n', '\r\n'))
    policy = CryptPolicy.from_path(path)
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    uc2 = to_bytes(self.sample_config_1s, 'utf-16', source_encoding='utf-8')
    set_file(path, uc2)
    policy = CryptPolicy.from_path(path, encoding='utf-16')
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)