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
def filter_on_attributes(ava, required=None, optional=None, acs=None, fail_on_unfulfilled_requirements=True):
    """Filter

    :param ava: An attribute value assertion as a dictionary
    :param required: list of RequestedAttribute instances defined to be
        required
    :param optional: list of RequestedAttribute instances defined to be
        optional
    :param fail_on_unfulfilled_requirements: If required attributes
        are missing fail or fail not depending on this parameter.
    :return: The modified attribute value assertion
    """

    def _match_attr_name(attr, ava):
        name = attr['name'].lower()
        name_format = attr.get('name_format')
        friendly_name = attr.get('friendly_name')
        local_name = get_local_name(acs, name, name_format) or friendly_name or ''
        _fn = _match(local_name, ava) or _match(name, ava)
        return _fn

    def _apply_attr_value_restrictions(attr, res, must=False):
        values = [av['text'] for av in attr.get('attribute_value', [])]
        try:
            res[_fn].extend(_filter_values(ava[_fn], values))
        except KeyError:
            val = _filter_values(ava[_fn], values)
            res[_fn] = val if val is not None else []
        return _filter_values(ava[_fn], values, must)
    res = {}
    if required is None:
        required = []
    for attr in required:
        _fn = _match_attr_name(attr, ava)
        if _fn:
            _apply_attr_value_restrictions(attr, res, True)
        elif fail_on_unfulfilled_requirements:
            desc = f"Required attribute missing: '{attr['name']}'"
            raise MissingValue(desc)
    if optional is None:
        optional = []
    for attr in optional:
        _fn = _match_attr_name(attr, ava)
        if _fn:
            _apply_attr_value_restrictions(attr, res, False)
    return res