from __future__ import with_statement
from passlib.utils.compat import PY3
import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, LazyCryptContext
from passlib.exc import PasslibConfigWarning, PasslibHashWarning
from passlib.utils import tick, to_unicode
from passlib.utils.compat import irange, u, unicode, str_to_uascii, PY2, PY26
import passlib.utils.handlers as uh
from passlib.tests.utils import (TestCase, set_file, TICK_RESOLUTION,
from passlib.registry import (register_crypt_handler_path,
import hashlib, time
def test_51_linear_vary_rounds(self):
    """test linear vary rounds"""
    cc = CryptContext(schemes=['sha256_crypt'], sha256_crypt__min_rounds=1995, sha256_crypt__max_rounds=2005, sha256_crypt__default_rounds=2000)
    self.assertRaises(ValueError, cc.copy, all__vary_rounds=-1)
    self.assertRaises(ValueError, cc.copy, all__vary_rounds='-1%')
    self.assertRaises(ValueError, cc.copy, all__vary_rounds='101%')
    c2 = cc.copy(all__vary_rounds=0)
    self.assertEqual(c2._get_record('sha256_crypt', None).vary_rounds, 0)
    self.assert_rounds_range(c2, 'sha256_crypt', 2000, 2000)
    c2 = cc.copy(all__vary_rounds='0%')
    self.assertEqual(c2._get_record('sha256_crypt', None).vary_rounds, 0)
    self.assert_rounds_range(c2, 'sha256_crypt', 2000, 2000)
    c2 = cc.copy(all__vary_rounds=1)
    self.assertEqual(c2._get_record('sha256_crypt', None).vary_rounds, 1)
    self.assert_rounds_range(c2, 'sha256_crypt', 1999, 2001)
    c2 = cc.copy(all__vary_rounds=100)
    self.assertEqual(c2._get_record('sha256_crypt', None).vary_rounds, 100)
    self.assert_rounds_range(c2, 'sha256_crypt', 1995, 2005)
    c2 = cc.copy(all__vary_rounds='0.1%')
    self.assertEqual(c2._get_record('sha256_crypt', None).vary_rounds, 0.001)
    self.assert_rounds_range(c2, 'sha256_crypt', 1998, 2002)
    c2 = cc.copy(all__vary_rounds='100%')
    self.assertEqual(c2._get_record('sha256_crypt', None).vary_rounds, 1.0)
    self.assert_rounds_range(c2, 'sha256_crypt', 1995, 2005)