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
def test_from_source(self):
    """from_source()"""
    from passlib.totp import TOTP
    from_source = TOTP.from_source
    otp = from_source(u('otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&issuer=Example'))
    self.assertEqual(otp.key, KEY4_RAW)
    otp = from_source(b'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&issuer=Example')
    self.assertEqual(otp.key, KEY4_RAW)
    otp = from_source(dict(v=1, type='totp', key=KEY4))
    self.assertEqual(otp.key, KEY4_RAW)
    otp = from_source(u('{"v": 1, "type": "totp", "key": "JBSWY3DPEHPK3PXP"}'))
    self.assertEqual(otp.key, KEY4_RAW)
    otp = from_source(b'{"v": 1, "type": "totp", "key": "JBSWY3DPEHPK3PXP"}')
    self.assertEqual(otp.key, KEY4_RAW)
    self.assertIs(from_source(otp), otp)
    wallet1 = AppWallet()
    otp1 = TOTP.using(wallet=wallet1).from_source(otp)
    self.assertIsNot(otp1, otp)
    self.assertEqual(otp1.to_dict(), otp.to_dict())
    otp2 = TOTP.using(wallet=wallet1).from_source(otp1)
    self.assertIs(otp2, otp1)
    self.assertRaises(ValueError, from_source, u('foo'))
    self.assertRaises(ValueError, from_source, b'foo')