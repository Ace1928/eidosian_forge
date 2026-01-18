from breezy.bzr import inventory, inventorytree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_extract_while_locked(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree(['subtree/'])
    tree.add(['subtree'])
    subtree = tree.extract('subtree')