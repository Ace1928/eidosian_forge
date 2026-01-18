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
def test_10_load(self):
    """test load() / load_path() method"""
    ctx = CryptContext()
    ctx.load(self.sample_1_dict)
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    ctx.load(self.sample_1_unicode)
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    ctx.load(self.sample_1_unicode.encode('utf-8'))
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    self.assertRaises(TypeError, ctx.load, None)
    ctx = CryptContext(**self.sample_1_dict)
    ctx.load({}, update=True)
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    ctx = CryptContext()
    ctx.load(self.sample_1_dict)
    ctx.load(self.sample_2_dict)
    self.assertEqual(ctx.to_dict(), self.sample_2_dict)