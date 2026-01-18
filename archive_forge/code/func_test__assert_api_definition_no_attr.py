from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test__assert_api_definition_no_attr(self):
    self.assertRaises(NotImplementedError, self.extn._assert_api_definition, attr='NOPE')