from breezy import osutils
from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
def test_list_files_with_unfollowed_reference(self):
    tree, subtree = self.create_nested()
    expected = [('', 'V', 'directory'), ('subtree', 'V', 'tree-reference')]
    self.assertFilesListEqual(tree, expected, recursive=True, recurse_nested=False, include_root=True)