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
def test_06_forbidden(self):
    """test CryptPolicy() forbidden kwds"""
    self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], des_crypt__salt='xx')
    self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], all__salt='xx')
    self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], user__context__schemes=['md5_crypt'])