from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_merge_tags_selector(self):
    b1, [revid, revid1] = self.make_branch_with_revision_tuple('b1', 2)
    w2 = b1.controldir.sprout('b2', revision_id=revid).open_workingtree()
    revid2 = w2.commit('revision 2')
    b2 = w2.branch
    b1.tags.set_tag('tag1', revid)
    b1.tags.set_tag('tag2', revid2)
    updates, conflicts = b1.tags.merge_to(b2.tags, selector=lambda x: x == 'tag1')
    self.assertEqual({'tag1': revid}, updates)
    self.assertEqual(set(), set(conflicts))
    self.assertEqual(b2.tags.lookup_tag('tag1'), revid)
    self.assertRaises(errors.NoSuchTag, b2.tags.lookup_tag, 'tag2')