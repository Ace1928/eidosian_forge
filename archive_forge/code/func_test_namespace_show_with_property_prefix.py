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
def test_namespace_show_with_property_prefix(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    rt = resource_types.ResourceTypeAssociation()
    rt.name = RESOURCE_TYPE2
    rt.prefix = 'pref'
    rt = self.rt_controller.create(request, rt, NAMESPACE3)
    object = objects.MetadefObject()
    object.name = OBJECT3
    object.required = []
    property = properties.PropertyType()
    property.name = PROPERTY2
    property.type = 'string'
    property.title = 'title'
    object.properties = {'prop1': property}
    object = self.object_controller.create(request, object, NAMESPACE3)
    self.assertNotificationsLog([{'type': 'metadef_resource_type.create', 'payload': {'namespace': NAMESPACE3, 'name': RESOURCE_TYPE2, 'prefix': 'pref', 'properties_target': None}}, {'type': 'metadef_object.create', 'payload': {'name': OBJECT3, 'namespace': NAMESPACE3, 'properties': [{'name': 'prop1', 'additionalItems': None, 'confidential': None, 'title': 'title', 'default': None, 'pattern': None, 'enum': None, 'maximum': None, 'minItems': None, 'minimum': None, 'maxItems': None, 'minLength': None, 'uniqueItems': None, 'maxLength': None, 'items': None, 'type': 'string', 'description': None}], 'required': [], 'description': None}}])
    filters = {'resource_type': RESOURCE_TYPE2}
    output = self.namespace_controller.show(request, NAMESPACE3, filters)
    output = output.to_dict()
    [self.assertTrue(property_name.startswith(rt.prefix)) for property_name in output['properties'].keys()]
    for object in output['objects']:
        [self.assertTrue(property_name.startswith(rt.prefix)) for property_name in object.properties.keys()]