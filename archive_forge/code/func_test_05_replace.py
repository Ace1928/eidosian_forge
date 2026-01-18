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
def test_05_replace(self):
    """test CryptPolicy.replace() constructor"""
    p1 = CryptPolicy(**self.sample_config_1pd)
    p2 = p1.replace(**self.sample_config_2pd)
    self.assertEqual(p2.to_dict(), self.sample_config_12pd)
    p2b = p2.replace(**self.sample_config_2pd)
    self.assertEqual(p2b.to_dict(), self.sample_config_12pd)
    p3 = p2.replace(self.sample_config_3pd)
    self.assertEqual(p3.to_dict(), self.sample_config_123pd)