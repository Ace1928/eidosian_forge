from breezy.bzr import inventory, inventorytree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def prepare_with_subtree(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    subtree = self.make_branch_and_tree('subtree')
    subtree.commit('dummy')
    tree.add(['subtree'])
    return tree