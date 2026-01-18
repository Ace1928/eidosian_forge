from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_merge_tags(self):
    b1, [revid, revid1] = self.make_branch_with_revision_tuple('b1', 2)
    w2 = b1.controldir.sprout('b2', revision_id=revid).open_workingtree()
    revid2 = w2.commit('revision 2')
    b2 = w2.branch
    b1.tags.set_tag('tagname', revid)
    b1.tags.merge_to(b2.tags)
    self.assertEqual(b2.tags.lookup_tag('tagname'), revid)
    b2.tags.set_tag('in-destination', revid)
    updates, conflicts = b1.tags.merge_to(b2.tags)
    self.assertEqual(list(conflicts), [])
    self.assertEqual(updates, {})
    self.assertEqual(b2.tags.lookup_tag('in-destination'), revid)
    b1.tags.set_tag('conflicts', revid1)
    b2.tags.set_tag('conflicts', revid2)
    updates, conflicts = b1.tags.merge_to(b2.tags)
    self.assertEqual(list(conflicts), [('conflicts', revid1, revid2)])
    self.assertEqual(updates, {})
    self.assertEqual(b2.tags.lookup_tag('conflicts'), revid2)