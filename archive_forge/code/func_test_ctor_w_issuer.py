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
def test_ctor_w_issuer(self):
    """constructor -- 'issuer' parameter"""
    self.assertEqual(TOTP(KEY1).issuer, None)
    self.assertEqual(TOTP(KEY1, issuer='foo.com').issuer, 'foo.com')
    self.assertRaises(ValueError, TOTP, KEY1, issuer='foo.com:bar')