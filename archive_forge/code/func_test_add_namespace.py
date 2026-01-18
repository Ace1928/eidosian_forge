from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_add_namespace(self):
    namespace = _db_namespace_fixture(namespace='added_namespace', display_name='fake', description='fake_desc', visibility='public', protected=True, owner=TENANT1)
    self.assertEqual('added_namespace', namespace['namespace'])
    self.db.metadef_namespace_create(None, namespace)
    retrieved_namespace = self.namespace_repo.get(namespace['namespace'])
    self.assertEqual('added_namespace', retrieved_namespace.namespace)