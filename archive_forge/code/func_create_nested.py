from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def create_nested(self):
    work_tree = self.make_branch_and_tree('wt')
    with work_tree.lock_write():
        self.skip_if_no_reference(work_tree)
        subtree = self.make_branch_and_tree('wt/subtree')
        subtree.commit('foo')
        work_tree.add_reference(subtree)
    tree = self._convert_tree(work_tree)
    self.skip_if_no_reference(tree)
    return (tree, subtree)