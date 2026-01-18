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
def test_11_iter_handlers(self):
    """test iter_handlers() method"""
    p1 = CryptPolicy(**self.sample_config_1pd)
    s = self.sample_config_1prd['schemes']
    self.assertEqual(list(p1.iter_handlers()), s)
    p3 = CryptPolicy(**self.sample_config_3pd)
    self.assertEqual(list(p3.iter_handlers()), [])