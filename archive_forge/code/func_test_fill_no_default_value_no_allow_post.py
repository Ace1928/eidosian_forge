from oslo_utils import uuidutils
import testtools
from webob import exc
from neutron_lib.api import attributes
from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib import constants
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
def test_fill_no_default_value_no_allow_post(self):
    attr_info = {'key': {'allow_post': False}}
    attr_inst = attributes.AttributeInfo(attr_info)
    self.assertRaises(exceptions.InvalidInput, self._test_fill_default_value, attr_inst, {'key': 'X'}, {'key': 'X'})
    self._test_fill_default_value(attr_inst, {}, {})
    self.assertRaises(self._EXC_CLS, attr_inst.fill_post_defaults, {'key': 'X'}, self._EXC_CLS)