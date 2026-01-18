from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_delete_nonexistant_extra_property(self):
    extra_properties = {}
    roles = ['spl_role']
    extra_prop_proxy = property_protections.ExtraPropertiesProxy(roles, extra_properties, self.property_rules)
    self.assertRaises(KeyError, extra_prop_proxy.__delitem__, 'spl_read_prop')