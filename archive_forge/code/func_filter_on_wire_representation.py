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
def filter_on_wire_representation(ava, acs, required=None, optional=None):
    """
    :param ava: A dictionary with attributes and values
    :param acs: List of tuples (Attribute Converter name,
        Attribute Converter instance)
    :param required: A list of saml.Attributes
    :param optional: A list of saml.Attributes
    :return: Dictionary of expected/wanted attributes and values
    """
    acsdic = {ac.name_format: ac for ac in acs}
    if required is None:
        required = []
    if optional is None:
        optional = []
    res = {}
    for attr, val in ava.items():
        done = False
        for req in required:
            try:
                _name = acsdic[req.name_format]._to[attr]
                if _name == req.name:
                    res[attr] = val
                    done = True
            except KeyError:
                pass
        if done:
            continue
        for opt in optional:
            try:
                _name = acsdic[opt.name_format]._to[attr]
                if _name == opt.name:
                    res[attr] = val
                    break
            except KeyError:
                pass
    return res