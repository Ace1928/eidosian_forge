from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_namespace_not_found(self):
    fake_namespace = 'fake_namespace'
    exc = self.assertRaises(exception.NotFound, self.namespace_repo.get, fake_namespace)
    self.assertIn(fake_namespace, encodeutils.exception_to_unicode(exc))