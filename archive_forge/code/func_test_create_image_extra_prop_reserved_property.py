from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_create_image_extra_prop_reserved_property(self):
    self.context = glance.context.RequestContext(tenant=TENANT1, roles=['spl_role'])
    self.image_factory = property_protections.ProtectedImageFactoryProxy(self.factory, self.context, self.property_rules)
    extra_props = {'foo': 'bar', 'spl_create_prop': 'c'}
    self.assertRaises(exception.ReservedProperty, self.image_factory.new_image, extra_properties=extra_props)