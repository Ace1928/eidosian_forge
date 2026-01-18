from breezy import errors
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
def test_mainline_ghost(self):
    tree = self.make_branch_and_tree('tree1')
    if not tree.branch.repository._format.supports_ghosts:
        raise TestNotApplicable('repository format does not support ghosts')
    tree.set_parent_ids([b'spooky'], allow_leftmost_as_ghost=True)
    tree.add('')
    tree.commit('msg1', rev_id=b'rev1')
    tree.commit('msg2', rev_id=b'rev2')
    self.assertRaises((errors.NoSuchRevision, errors.GhostRevisionsHaveNoRevno), tree.branch.revision_id_to_revno, b'unknown')
    self.assertEqual(1, tree.branch.revision_id_to_revno(b'rev1'))
    self.assertEqual(2, tree.branch.revision_id_to_revno(b'rev2'))