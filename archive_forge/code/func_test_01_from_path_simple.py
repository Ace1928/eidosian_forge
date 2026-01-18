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
def test_01_from_path_simple(self):
    """test CryptPolicy.from_path() constructor"""
    path = self.sample_config_1s_path
    policy = CryptPolicy.from_path(path)
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    self.assertRaises(EnvironmentError, CryptPolicy.from_path, path + 'xxx')