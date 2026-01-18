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
def test_convert_value(self):
    attr_info = {'key': {}}
    attr_inst = attributes.AttributeInfo(attr_info)
    self._test_convert_value(attr_inst, {'key': constants.ATTR_NOT_SPECIFIED}, {'key': constants.ATTR_NOT_SPECIFIED})
    self._test_convert_value(attr_inst, {'key': 'X'}, {'key': 'X'})
    self._test_convert_value(attr_inst, {'other_key': 'X'}, {'other_key': 'X'})
    attr_info = {'key': {'convert_to': converters.convert_to_int}}
    attr_inst = attributes.AttributeInfo(attr_info)
    self._test_convert_value(attr_inst, {'key': constants.ATTR_NOT_SPECIFIED}, {'key': constants.ATTR_NOT_SPECIFIED})
    self._test_convert_value(attr_inst, {'key': 1}, {'key': '1'})
    self._test_convert_value(attr_inst, {'key': 1}, {'key': 1})
    self.assertRaises(exceptions.InvalidInput, self._test_convert_value, attr_inst, {'key': 1}, {'key': 'a'})
    attr_info = {'key': {'validate': {'type:uuid': None}}}
    attr_inst = attributes.AttributeInfo(attr_info)
    self._test_convert_value(attr_inst, {'key': constants.ATTR_NOT_SPECIFIED}, {'key': constants.ATTR_NOT_SPECIFIED})
    uuid_str = '01234567-1234-1234-1234-1234567890ab'
    self._test_convert_value(attr_inst, {'key': uuid_str}, {'key': uuid_str})
    self.assertRaises(exceptions.InvalidInput, self._test_convert_value, attr_inst, {'key': 1}, {'key': 1})
    self.assertRaises(self._EXC_CLS, attr_inst.convert_values, {'key': 1}, self._EXC_CLS)