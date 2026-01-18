from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_add_tag_namespace_forbidden(self):
    tag = _db_tag_fixture(name='added_tag')
    self.assertEqual('added_tag', tag['name'])
    self.assertRaises(exception.Forbidden, self.db.metadef_tag_create, self.context, NAMESPACE3, tag)