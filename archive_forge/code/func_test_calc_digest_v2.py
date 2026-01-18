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
def test_calc_digest_v2(self):
    """
        test digest calc v2 matches bcrypt()
        """
    from passlib.hash import bcrypt
    from passlib.crypto.digest import compile_hmac
    from passlib.utils.binary import b64encode
    salt = 'nyKYxTAvjmy6lMDYMl11Uu'
    secret = 'test'
    temp_digest = compile_hmac('sha256', salt.encode('ascii'))(secret.encode('ascii'))
    temp_digest = b64encode(temp_digest).decode('ascii')
    self.assertEqual(temp_digest, 'J5TlyIDm+IcSWmKiDJm+MeICndBkFVPn4kKdJW8f+xY=')
    bcrypt_digest = bcrypt(ident='2b', salt=salt, rounds=12)._calc_checksum(temp_digest)
    self.assertEqual(bcrypt_digest, 'M0wE0Ov/9LXoQFCe.jRHu3MSHPF54Ta')
    self.assertTrue(bcrypt.verify(temp_digest, '$2b$12$' + salt + bcrypt_digest))
    result = self.handler(ident='2b', salt=salt, rounds=12)._calc_checksum(secret)
    self.assertEqual(result, bcrypt_digest)