from breezy import errors, tests
from breezy import transport as _mod_transport
from breezy.tests import per_workingtree
def test_get_pending_merge_revision_tree(self):
    tree = self.make_branch_and_tree('tree1')
    tree.commit('first post')
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    revision1 = tree2.commit('commit in branch', allow_pointless=True)
    tree.merge_from_branch(tree2.branch)
    try:
        cached_revision_tree = tree.revision_tree(revision1)
    except errors.NoSuchRevision:
        return
    real_revision_tree = tree2.basis_tree()
    self.assertTreesEqual(real_revision_tree, cached_revision_tree)