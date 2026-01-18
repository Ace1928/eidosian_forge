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
def do_attribute(val, typ, key):
    attr = saml.Attribute()
    attrval = do_ava(val, typ)
    if attrval:
        attr.attribute_value = attrval
    if isinstance(key, str):
        attr.name = key
    elif isinstance(key, tuple):
        try:
            name, nformat, friendly = key
        except ValueError:
            name, nformat = key
            friendly = ''
        if name:
            attr.name = name
        if nformat:
            attr.name_format = nformat
        if friendly:
            attr.friendly_name = friendly
    return attr