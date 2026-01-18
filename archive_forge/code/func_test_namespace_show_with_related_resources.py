import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_namespace_show_with_related_resources(self):
    request = unit_test_utils.get_fake_request()
    output = self.namespace_controller.show(request, NAMESPACE3)
    output = output.to_dict()
    self.assertEqual(NAMESPACE3, output['namespace'])
    self.assertEqual(TENANT3, output['owner'])
    self.assertFalse(output['protected'])
    self.assertEqual('public', output['visibility'])
    self.assertEqual(2, len(output['properties']))
    actual = set([property for property in output['properties']])
    expected = set([PROPERTY1, PROPERTY2])
    self.assertEqual(expected, actual)
    self.assertEqual(2, len(output['objects']))
    actual = set([object.name for object in output['objects']])
    expected = set([OBJECT1, OBJECT2])
    self.assertEqual(expected, actual)
    self.assertEqual(1, len(output['resource_type_associations']))
    actual = set([rt.name for rt in output['resource_type_associations']])
    expected = set([RESOURCE_TYPE1])
    self.assertEqual(expected, actual)