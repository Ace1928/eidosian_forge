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
def test_20_iter_config(self):
    """test iter_config() method"""
    p5 = CryptPolicy(**self.sample_config_5pd)
    self.assertEqual(dict(p5.iter_config()), self.sample_config_5pd)
    self.assertEqual(dict(p5.iter_config(resolve=True)), self.sample_config_5prd)
    self.assertEqual(dict(p5.iter_config(ini=True)), self.sample_config_5pid)