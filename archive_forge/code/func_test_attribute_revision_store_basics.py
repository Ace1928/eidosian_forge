from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_attribute_revision_store_basics(self):
    """Test the basic behaviour of the revisions attribute."""
    tree = self.make_branch_and_tree('tree')
    repo = tree.branch.repository
    repo.lock_write()
    try:
        self.assertEqual(set(), set(repo.revisions.keys()))
        revid = (tree.commit('foo'),)
        self.assertEqual({revid}, set(repo.revisions.keys()))
        self.assertEqual({revid: ()}, repo.revisions.get_parent_map([revid]))
    finally:
        repo.unlock()
    tree2 = self.make_branch_and_tree('tree2')
    tree2.pull(tree.branch)
    left_id = (tree2.commit('left'),)
    right_id = (tree.commit('right'),)
    tree.merge_from_branch(tree2.branch)
    merge_id = (tree.commit('merged'),)
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertEqual({revid, left_id, right_id, merge_id}, set(repo.revisions.keys()))
    self.assertEqual({revid: (), left_id: (revid,), right_id: (revid,), merge_id: (right_id, left_id)}, repo.revisions.get_parent_map(repo.revisions.keys()))