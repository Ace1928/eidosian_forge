from .. import missing, tests
from ..missing import iter_log_revisions
from . import TestCaseWithTransport
def test_merged(self):
    tree = self.make_branch_and_tree('tree')
    rev1 = tree.commit('one')
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    tree2.commit('two')
    tree2.commit('three')
    tree.merge_from_branch(tree2.branch)
    rev4 = tree.commit('four')
    self.assertUnmerged([('2', rev4, 0)], [], tree.branch, tree2.branch)
    self.assertUnmerged([('2', rev4, 0)], [], tree.branch, tree2.branch, local_revid_range=(rev4, rev4))
    self.assertUnmerged([], [], tree.branch, tree2.branch, local_revid_range=(rev1, rev1))