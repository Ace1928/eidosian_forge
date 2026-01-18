import base64
import hashlib
import hmac
import logging
import random
import string
import sys
import traceback
import zlib
from saml2 import VERSION
from saml2 import saml
from saml2 import samlp
from saml2.time_util import instant
def rndbytes(size=16, alphabet=''):
    """
    Returns rndstr always as a binary type
    """
    x = rndstr(size, alphabet)
    if isinstance(x, str):
        return x.encode('utf-8')
    return x