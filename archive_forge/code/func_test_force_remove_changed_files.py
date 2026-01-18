from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_force_remove_changed_files(self):
    """Check that changed files are removed and deleted when forced."""
    tree = self.get_tree(TestRemove.files)
    tree.add(TestRemove.files)
    tree.remove(TestRemove.files, keep_files=False, force=True)
    self.assertRemovedAndDeleted(TestRemove.files)
    self.assertPathDoesNotExist(['a.~1~', 'b.~1~/', 'b.~1~/c', 'd.~1~/'])
    tree._validate()