from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_fail_if_missing(self):
    """We should not be able to bind to a missing branch."""
    tree = self.make_branch_and_tree('tree_1')
    tree.commit('dummy commit')
    self.run_bzr_error(['Not a branch.*no-such-branch/'], ['bind', '../no-such-branch'], working_dir='tree_1')
    self.assertIs(None, tree.branch.get_bound_location())