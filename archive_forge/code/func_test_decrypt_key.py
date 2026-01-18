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
def test_decrypt_key(self):
    """.decrypt_key()"""
    wallet = AppWallet({'1': PASS1, '2': PASS2})
    CIPHER1 = dict(v=1, c=13, s='6D7N7W53O7HHS37NLUFQ', k='MHCTEGSNPFN5CGBJ', t='1')
    self.require_aes_support(canary=partial(wallet.decrypt_key, CIPHER1))
    self.assertEqual(wallet.decrypt_key(CIPHER1)[0], KEY1_RAW)
    CIPHER2 = dict(v=1, c=13, s='SPZJ54Y6IPUD2BYA4C6A', k='ZGDXXTVQOWYLC2AU', t='1')
    self.assertEqual(wallet.decrypt_key(CIPHER2)[0], KEY1_RAW)
    CIPHER3 = dict(v=1, c=8, s='FCCTARTIJWE7CPQHUDKA', k='D2DRS32YESGHHINWFFCELKN7Z6NAHM4M', t='2')
    self.assertEqual(wallet.decrypt_key(CIPHER3)[0], KEY2_RAW)
    temp = CIPHER1.copy()
    temp.update(t='2')
    self.assertEqual(wallet.decrypt_key(temp)[0], b'\xafD6.F7\xeb\x19\x05Q')
    temp = CIPHER1.copy()
    temp.update(t='3')
    self.assertRaises(KeyError, wallet.decrypt_key, temp)
    temp = CIPHER1.copy()
    temp.update(v=999)
    self.assertRaises(ValueError, wallet.decrypt_key, temp)