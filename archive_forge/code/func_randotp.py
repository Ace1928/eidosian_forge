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
def randotp(self, cls=None, **kwds):
    """
        helper which generates a random TOTP instance.
        """
    rng = self.getRandom()
    if 'key' not in kwds:
        kwds['new'] = True
    kwds.setdefault('digits', rng.randint(6, 10))
    kwds.setdefault('alg', rng.choice(['sha1', 'sha256', 'sha512']))
    kwds.setdefault('period', rng.randint(10, 120))
    return (cls or TOTP)(**kwds)