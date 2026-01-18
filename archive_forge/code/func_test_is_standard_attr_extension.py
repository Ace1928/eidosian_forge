import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_is_standard_attr_extension(self):
    if self.is_standard_attr_extension:
        self.assertIn('standard-attr-', self.alias)
    else:
        self.skipTest('API definition is not related to standardattr.')