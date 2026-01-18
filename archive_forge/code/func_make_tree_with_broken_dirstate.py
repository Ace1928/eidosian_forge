from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def make_tree_with_broken_dirstate(self, path):
    tree = self.make_branch_and_tree(path)
    self.break_dirstate(tree)
    return tree