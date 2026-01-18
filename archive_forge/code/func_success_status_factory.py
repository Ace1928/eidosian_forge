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
def success_status_factory():
    return samlp.Status(status_code=samlp.StatusCode(value=samlp.STATUS_SUCCESS))