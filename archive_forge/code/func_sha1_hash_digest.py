from __future__ import print_function
import base64
import hashlib
import os
from cStringIO import StringIO
from M2Crypto import BIO, EVP, RSA, X509, m2
def sha1_hash_digest(payload):
    """Create a SHA1 hash and return the base64 string"""
    return base64.b64encode(hashlib.sha1(payload).digest())