import unittest
import binascii
from Cryptodome.Util.py3compat import b
def strip_whitespace(s):
    """Remove whitespace from a text or byte string"""
    if isinstance(s, str):
        return b(''.join(s.split()))
    else:
        return b('').join(s.split())