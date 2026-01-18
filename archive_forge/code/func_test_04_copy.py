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
def test_04_copy(self):
    """test copy() method"""
    cc1 = CryptContext(**self.sample_1_dict)
    cc2 = cc1.copy(**self.sample_2_dict)
    self.assertEqual(cc1.to_dict(), self.sample_1_dict)
    self.assertEqual(cc2.to_dict(), self.sample_12_dict)
    cc2b = cc2.copy(**self.sample_2_dict)
    self.assertEqual(cc1.to_dict(), self.sample_1_dict)
    self.assertEqual(cc2b.to_dict(), self.sample_12_dict)
    cc3 = cc2.copy(**self.sample_3_dict)
    self.assertEqual(cc3.to_dict(), self.sample_123_dict)
    cc4 = cc1.copy()
    self.assertIsNot(cc4, cc1)
    self.assertEqual(cc1.to_dict(), self.sample_1_dict)
    self.assertEqual(cc4.to_dict(), self.sample_1_dict)
    cc4.update(**self.sample_2_dict)
    self.assertEqual(cc1.to_dict(), self.sample_1_dict)
    self.assertEqual(cc4.to_dict(), self.sample_12_dict)