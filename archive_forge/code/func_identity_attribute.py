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
def identity_attribute(form, attribute, forward_map=None):
    if form == 'friendly':
        if attribute.friendly_name:
            return attribute.friendly_name
        elif forward_map:
            try:
                return forward_map[attribute.name, attribute.name_format]
            except KeyError:
                return attribute.name
    return attribute.name