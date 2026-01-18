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
def parse_attribute_map(filenames):
    """
    Expects a file with each line being composed of the oid for the attribute
    exactly one space, a user friendly name of the attribute and then
    the type specification of the name.

    :param filenames: List of filenames on mapfiles.
    :return: A 2-tuple, one dictionary with the oid as keys and the friendly
        names as values, the other one the other way around.
    """
    forward = {}
    backward = {}
    for filename in filenames:
        with open(filename) as fp:
            for line in fp:
                name, friendly_name, name_format = line.strip().split()
                forward[name, name_format] = friendly_name
                backward[friendly_name] = (name, name_format)
    return (forward, backward)