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
def test_match_w_window(self):
    """match() -- 'time' and 'window' parameters"""
    otp = self.randotp()
    period = otp.period
    time = self.randtime()
    token = otp.generate(time).token
    common = dict(otp=otp, gen_time=time)
    assertMatches = partial(self.assertVerifyMatches, **common)
    assertRaises = partial(self.assertVerifyRaises, **common)
    assertRaises(exc.InvalidTokenError, token, time - period, window=0)
    assertMatches(+1, token, time - period, window=period)
    assertMatches(+1, token, time - period, window=2 * period)
    assertMatches(0, token, time, window=0)
    assertRaises(exc.InvalidTokenError, token, time + period, window=0)
    assertMatches(-1, token, time + period, window=period)
    assertMatches(-1, token, time + period, window=2 * period)
    assertRaises(exc.InvalidTokenError, token, time + 2 * period, window=0)
    assertRaises(exc.InvalidTokenError, token, time + 2 * period, window=period)
    assertMatches(-2, token, time + 2 * period, window=2 * period)
    dt = datetime.datetime.utcfromtimestamp(time)
    assertMatches(0, token, dt, window=0)
    assertRaises(ValueError, token, -1)