import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def assertParams(self, resource):
    for attribute in resource.keys():
        self.assertIn(attribute, base.KNOWN_ATTRIBUTES + self.extension_attributes, 'Attribute is unknown, check for typos.')
        for keyword in resource[attribute]:
            self.assertIn(keyword, base.KNOWN_KEYWORDS, 'Keyword is unknown, check for typos.')
            value = resource[attribute][keyword]
            assert_f = ASSERT_FUNCTIONS[keyword]
            assert_f(self, attribute, resource[attribute], keyword, value)