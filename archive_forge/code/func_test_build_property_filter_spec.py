import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_build_property_filter_spec(self):
    client_factory = mock.Mock()
    prop_specs = [mock.Mock()]
    obj_specs = [mock.Mock()]
    filter_spec = vim_util.build_property_filter_spec(client_factory, prop_specs, obj_specs)
    self.assertEqual(obj_specs, filter_spec.objectSet)
    self.assertEqual(prop_specs, filter_spec.propSet)