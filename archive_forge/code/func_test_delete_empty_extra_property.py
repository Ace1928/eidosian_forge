from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_delete_empty_extra_property(self):
    extra_properties = {'foo': ''}
    context = glance.context.RequestContext(roles=['admin'])
    extra_prop_proxy = property_protections.ExtraPropertiesProxy(context, extra_properties, self.property_rules)
    del extra_prop_proxy['foo']
    self.assertNotIn('foo', extra_prop_proxy)