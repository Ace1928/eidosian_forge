from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.handlers.bcrypt import IDENT_2, IDENT_2X
from passlib.utils import repeat_string, to_bytes, is_safe_crypt_input
from passlib.utils.compat import irange, PY3
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE
def check_pybcrypt(secret, hash):
    """pybcrypt"""
    secret = to_native_str(secret, self.FuzzHashGenerator.password_encoding)
    if len(secret) > 200:
        secret = secret[:200]
    if hash.startswith((IDENT_2B, IDENT_2Y)):
        hash = IDENT_2A + hash[4:]
    try:
        if lock:
            with lock:
                return bcrypt_mod.hashpw(secret, hash) == hash
        else:
            return bcrypt_mod.hashpw(secret, hash) == hash
    except ValueError:
        raise ValueError('py-bcrypt rejected hash: %r' % (hash,))