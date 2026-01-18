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
def test_create_from_attribute_info_instance(self):
    cloned_attrs = attributes.AttributeInfo(TestAttributeInfo._ATTRS_INSTANCE)
    self.assertEqual(TestAttributeInfo._ATTRS_INSTANCE.attributes, cloned_attrs.attributes)