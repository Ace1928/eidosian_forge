from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_property(self):
    property = self.property_repo.get(NAMESPACE1, PROPERTY1)
    namespace = self.namespace_repo.get(NAMESPACE1)
    self.assertEqual(PROPERTY1, property.name)
    self.assertEqual(namespace.namespace, property.namespace.namespace)