from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_tag_empty_result(self):
    tags = self.tag_repo.list(filters={'namespace': NAMESPACE2})
    tag_names = set([t.name for t in tags])
    self.assertEqual(set([]), tag_names)