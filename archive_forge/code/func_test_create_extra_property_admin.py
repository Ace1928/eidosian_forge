from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_create_extra_property_admin(self):
    extra_properties = {}
    context = glance.context.RequestContext(roles=['admin'])
    extra_prop_proxy = property_protections.ExtraPropertiesProxy(context, extra_properties, self.property_rules)
    extra_prop_proxy['boo'] = 'doo'
    self.assertEqual('doo', extra_prop_proxy['boo'])