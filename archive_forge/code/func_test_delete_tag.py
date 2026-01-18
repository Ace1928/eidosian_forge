from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_delete_tag(self):
    tag_name = 'α'
    b, [revid] = self.make_branch_with_revision_tuple('b', 1)
    b.tags.set_tag(tag_name, revid)
    b.tags.delete_tag(tag_name)
    self.assertRaises(errors.NoSuchTag, b.tags.lookup_tag, tag_name)
    self.assertEqual(b.tags.get_tag_dict(), {})
    self.assertRaises(errors.NoSuchTag, b.tags.delete_tag, tag_name)
    self.assertRaises(errors.NoSuchTag, b.tags.delete_tag, tag_name + '2')