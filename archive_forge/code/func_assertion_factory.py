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
def assertion_factory(**kwargs):
    assertion = saml.Assertion(version=VERSION, id=sid(), issue_instant=instant())
    for key, val in kwargs.items():
        setattr(assertion, key, val)
    return assertion