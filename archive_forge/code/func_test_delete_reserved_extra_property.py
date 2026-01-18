from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_delete_reserved_extra_property(self):
    extra_properties = {'spl_read_prop': 'r'}
    context = glance.context.RequestContext(roles=['spl_role'])
    extra_prop_proxy = property_protections.ExtraPropertiesProxy(context, extra_properties, self.property_rules)
    self.assertEqual('r', extra_prop_proxy['spl_read_prop'])
    self.assertRaises(exception.ReservedProperty, extra_prop_proxy.__delitem__, 'spl_read_prop')