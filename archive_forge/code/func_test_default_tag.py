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
def test_default_tag(self):
    """constructor -- 'default_tag' param"""
    wallet = AppWallet({'1': 'one', '02': 'two'})
    self.assertEqual(wallet.default_tag, '02')
    self.assertEqual(wallet.get_secret(wallet.default_tag), b'two')
    wallet = AppWallet({'1': 'one', '02': 'two', 'A': 'aaa'})
    self.assertEqual(wallet.default_tag, 'A')
    self.assertEqual(wallet.get_secret(wallet.default_tag), b'aaa')
    wallet = AppWallet({'1': 'one', '02': 'two', 'A': 'aaa'}, default_tag='1')
    self.assertEqual(wallet.default_tag, '1')
    self.assertEqual(wallet.get_secret(wallet.default_tag), b'one')
    self.assertRaises(KeyError, AppWallet, {'1': 'one', '02': 'two', 'A': 'aaa'}, default_tag='B')
    wallet = AppWallet()
    self.assertEqual(wallet.default_tag, None)
    self.assertRaises(KeyError, wallet.get_secret, None)