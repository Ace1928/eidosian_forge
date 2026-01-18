from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_add_tags_with_append_true(self):
    tags = self.tag_repo.list(filters={'namespace': NAMESPACE1})
    tag_names = set([t.name for t in tags])
    self.assertEqual(set([TAG1, TAG2, TAG3]), tag_names)
    tags = _db_tags_fixture([TAG4, TAG5])
    self.db.metadef_tag_create_tags(self.context, NAMESPACE1, tags, can_append=True)
    tags = self.tag_repo.list(filters={'namespace': NAMESPACE1})
    tag_names = set([t.name for t in tags])
    self.assertEqual(set([TAG1, TAG2, TAG3, TAG4, TAG5]), tag_names)