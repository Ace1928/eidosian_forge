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
def test_encrypt_cost_timing(self):
    """verify cost parameter via timing"""
    self.require_aes_support()
    wallet = AppWallet({'1': 'aaa'})
    wallet.encrypt_cost -= 2
    delta, _ = time_call(partial(wallet.encrypt_key, KEY1_RAW), maxtime=0)
    wallet.encrypt_cost += 3
    delta2, _ = time_call(partial(wallet.encrypt_key, KEY1_RAW), maxtime=0)
    self.assertAlmostEqual(delta2, delta * 8, delta=delta * 8 * 0.5)