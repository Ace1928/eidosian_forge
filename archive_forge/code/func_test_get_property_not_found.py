from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_property_not_found(self):
    exc = self.assertRaises(exception.NotFound, self.property_repo.get, NAMESPACE2, PROPERTY1)
    self.assertIn(PROPERTY1, encodeutils.exception_to_unicode(exc))