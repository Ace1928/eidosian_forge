from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_one_new_id(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['one'])
    tree.add(['one'])
    self.assertTreeLayout(['', 'one'], tree)