import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def assert_converter(tester, attribute, attribute_dict, keyword, value):
    if 'default' not in attribute_dict or attribute_dict['default'] is constants.ATTR_NOT_SPECIFIED or attribute_dict.get(constants.DICT_POPULATE_DEFAULTS):
        return
    try:
        attribute_dict['convert_to'](attribute_dict['default'])
    except KeyError:
        try:
            attribute_dict['convert_list_to'](attribute_dict['default'])
        except KeyError:
            if validators.is_attr_set(value) and (not isinstance(value, (str, list))):
                tester.fail("Default value '%s' cannot be converted for attribute %s." % (value, attribute))