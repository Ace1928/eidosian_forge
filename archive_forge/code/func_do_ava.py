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
def do_ava(val, typ=''):
    if isinstance(val, str):
        ava = saml.AttributeValue()
        ava.set_text(val)
        attrval = [ava]
    elif isinstance(val, list):
        attrval = [do_ava(v)[0] for v in val]
    elif val or val is False:
        ava = saml.AttributeValue()
        ava.set_text(val)
        attrval = [ava]
    elif val is None:
        attrval = None
    else:
        raise OtherError(f'strange value type on: {val}')
    if typ:
        for ava in attrval:
            ava.set_type(typ)
    return attrval