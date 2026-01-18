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
def assertVerifyRaises(self, exc_class, token, time, otp, gen_time=None, **kwds):
    """helper to test otp.match() throws correct error"""
    msg = 'key=%r alg=%r period=%r token=%r gen_time=%r time=%r:' % (otp.base32_key, otp.alg, otp.period, token, gen_time, time)
    return self.assertRaises(exc_class, otp.match, token, time, __msg__=msg, **kwds)