from __future__ import annotations
import hashlib
import hmac
import os
import posixpath
import secrets
def gen_salt(length: int) -> str:
    """Generate a random string of SALT_CHARS with specified ``length``."""
    if length <= 0:
        raise ValueError('Salt length must be at least 1.')
    return ''.join((secrets.choice(SALT_CHARS) for _ in range(length)))