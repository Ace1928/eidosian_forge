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
def rndstr(size=16, alphabet=''):
    """
    Returns a string of random ascii characters or digits

    :param size: The length of the string
    :return: string
    """
    rng = random.SystemRandom()
    if not alphabet:
        alphabet = string.ascii_letters[0:52] + string.digits
    return type(alphabet)().join((rng.choice(alphabet) for _ in range(size)))