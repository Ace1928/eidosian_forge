from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
def has_native_md4():
    """
    check if hashlib natively supports md4.
    """
    try:
        hashlib.new('md4')
        return True
    except ValueError:
        return False