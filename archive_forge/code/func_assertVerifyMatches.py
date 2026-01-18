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
def assertVerifyMatches(self, expect_skipped, token, time, otp, gen_time=None, **kwds):
    """helper to test otp.match() output is correct"""
    msg = 'key=%r alg=%r period=%r token=%r gen_time=%r time=%r:' % (otp.base32_key, otp.alg, otp.period, token, gen_time, time)
    result = otp.match(token, time, **kwds)
    self.assertTotpMatch(result, time=otp.normalize_time(time), period=otp.period, window=kwds.get('window', 30), skipped=expect_skipped, msg=msg)