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
def iter_test_vectors(self):
    """
        helper to iterate over test vectors.
        yields ``(totp, time, token, expires, prefix)`` tuples.
        """
    from passlib.totp import TOTP
    for row in self.vectors:
        kwds = self.vector_defaults.copy()
        kwds.update(row[0])
        for entry in row[1:]:
            if len(entry) == 3:
                time, token, expires = entry
            else:
                time, token = entry
                expires = None
            log.debug('test vector: %r time=%r token=%r expires=%r', kwds, time, token, expires)
            otp = TOTP(**kwds)
            prefix = 'alg=%r time=%r token=%r: ' % (otp.alg, time, token)
            yield (otp, time, token, expires, prefix)