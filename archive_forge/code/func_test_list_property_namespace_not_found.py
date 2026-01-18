from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_property_namespace_not_found(self):
    exc = self.assertRaises(exception.NotFound, self.property_repo.list, filters={'namespace': 'not-a-namespace'})
    self.assertIn('not-a-namespace', encodeutils.exception_to_unicode(exc))