from breezy import osutils
from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
def test_list_files_with_followed_reference(self):
    tree, subtree = self.create_nested()
    expected = [('', 'V', 'directory'), ('subtree', 'V', 'directory'), ('subtree/a', 'V', 'file')]
    self.assertFilesListEqual(tree, expected, recursive=True, recurse_nested=True, include_root=True)