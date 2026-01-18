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
def decode_base64_and_inflate(string):
    """base64 decodes and then inflates according to RFC1951

    :param string: a deflated and encoded string
    :return: the string after decoding and inflating
    """
    return zlib.decompress(base64.b64decode(string), -15)