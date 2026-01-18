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
def status_message_factory(message, code, fro=samlp.STATUS_RESPONDER):
    return samlp.Status(status_message=samlp.StatusMessage(text=message), status_code=samlp.StatusCode(value=fro, status_code=samlp.StatusCode(value=code)))