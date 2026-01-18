import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import sys
import time as _time
from passlib import exc
from passlib.utils.compat import unicode, u
from passlib.tests.utils import TestCase, time_call
from passlib import totp as totp_module
from passlib.totp import TOTP, AppWallet, AES_SUPPORT
def test_ctor_w_alg(self):
    """constructor -- 'alg' parameter"""
    self.assertEqual(TOTP(KEY1, alg='SHA-256').alg, 'sha256')
    self.assertEqual(TOTP(KEY1, alg='SHA256').alg, 'sha256')
    self.assertRaises(ValueError, TOTP, KEY1, alg='SHA-333')