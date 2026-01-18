from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_multiple(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b', 'dir/', 'dir/subdir/', 'dir/subdir/foo'])
    tree.add(['a', 'b', 'dir', 'dir/subdir', 'dir/subdir/foo'])
    self.assertTreeLayout(['', 'a', 'b', 'dir/', 'dir/subdir/', 'dir/subdir/foo'], tree)