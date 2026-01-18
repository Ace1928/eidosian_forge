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
def test_harden_verify_parsing(self):
    """harden_verify -- parsing"""
    warnings.filterwarnings('ignore', '.*harden_verify.*', category=DeprecationWarning)
    ctx = CryptContext(schemes=['sha256_crypt'])
    self.assertEqual(ctx.harden_verify, None)
    self.assertEqual(ctx.using(harden_verify='').harden_verify, None)
    self.assertEqual(ctx.using(harden_verify='true').harden_verify, None)
    self.assertEqual(ctx.using(harden_verify='false').harden_verify, None)