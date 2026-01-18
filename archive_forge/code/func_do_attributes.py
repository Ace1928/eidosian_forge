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
def do_attributes(identity):
    attrs = []
    if not identity:
        return attrs
    for key, spec in identity.items():
        try:
            val, typ = spec
        except ValueError:
            val = spec
            typ = ''
        except TypeError:
            val = ''
            typ = ''
        attr = do_attribute(val, typ, key)
        attrs.append(attr)
    return attrs