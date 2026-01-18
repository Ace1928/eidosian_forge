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
def test_match_w_reference_vectors(self):
    """match() -- reference vectors"""
    for otp, time, token, expires, msg in self.iter_test_vectors():
        match = otp.match
        result = match(token, time)
        self.assertTrue(result)
        self.assertEqual(result.counter, time // otp.period, msg=msg)
        self.assertRaises(exc.InvalidTokenError, match, token, time + 100, window=0)