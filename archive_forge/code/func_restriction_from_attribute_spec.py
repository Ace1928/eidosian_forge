import copy
import importlib
import logging
import re
from warnings import warn as _warn
from saml2 import saml
from saml2 import xmlenc
from saml2.attribute_converter import ac_factory
from saml2.attribute_converter import from_local
from saml2.attribute_converter import get_local_name
from saml2.s_utils import MissingValue
from saml2.s_utils import assertion_factory
from saml2.s_utils import factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.time_util import in_a_while
from saml2.time_util import instant
def restriction_from_attribute_spec(attributes):
    restr = {}
    for attribute in attributes:
        restr[attribute.name] = {}
        for val in attribute.attribute_value:
            if not val.text:
                restr[attribute.name] = None
                break
            else:
                restr[attribute.name] = re.compile(val.text)
    return restr