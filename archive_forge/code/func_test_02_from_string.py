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
def test_02_from_string(self):
    """test CryptPolicy.from_string() constructor"""
    policy = CryptPolicy.from_string(self.sample_config_1s)
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    policy = CryptPolicy.from_string(self.sample_config_1s.replace('\n', '\r\n'))
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    data = to_unicode(self.sample_config_1s)
    policy = CryptPolicy.from_string(data)
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    uc2 = to_bytes(self.sample_config_1s, 'utf-16', source_encoding='utf-8')
    policy = CryptPolicy.from_string(uc2, encoding='utf-16')
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    policy = CryptPolicy.from_string(self.sample_config_4s)
    self.assertEqual(policy.to_dict(), self.sample_config_4pd)