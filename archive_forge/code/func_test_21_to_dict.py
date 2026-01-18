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
def test_21_to_dict(self):
    """test to_dict() method"""
    p5 = CryptPolicy(**self.sample_config_5pd)
    self.assertEqual(p5.to_dict(), self.sample_config_5pd)
    self.assertEqual(p5.to_dict(resolve=True), self.sample_config_5prd)