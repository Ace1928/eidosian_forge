from breezy.bzr import inventory, inventorytree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_iter_entries_by_dir_autodetects_subtree(self):
    tree = self.prepare_with_subtree()
    path, ie = next(tree.iter_entries_by_dir(specific_files=['subtree']))
    self.assertEqual('tree-reference', ie.kind)