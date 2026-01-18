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
def test_ctor_w_key_and_format(self):
    """constructor -- 'key' and 'format' parameters"""
    self.assertEqual(TOTP(KEY1).key, KEY1_RAW)
    self.assertEqual(TOTP(KEY1.lower()).key, KEY1_RAW)
    self.assertEqual(TOTP(' 4aog gdbb qsyh ntuz ').key, KEY1_RAW)
    self.assertRaises(Base32DecodeError, TOTP, 'ao!ggdbbqsyhntuz')
    self.assertEqual(TOTP('e01c630c2184b076ce99', 'hex').key, KEY1_RAW)
    self.assertRaises(Base16DecodeError, TOTP, 'X01c630c2184b076ce99', 'hex')
    self.assertEqual(TOTP(KEY1_RAW, 'raw').key, KEY1_RAW)