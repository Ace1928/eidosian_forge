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
def test_ctor_w_period(self):
    """constructor -- 'period' parameter"""
    self.assertEqual(TOTP(KEY1).period, 30)
    self.assertEqual(TOTP(KEY1, period=63).period, 63)
    self.assertRaises(TypeError, TOTP, KEY1, period=1.5)
    self.assertRaises(TypeError, TOTP, KEY1, period='abc')
    self.assertRaises(ValueError, TOTP, KEY1, period=0)
    self.assertRaises(ValueError, TOTP, KEY1, period=-1)