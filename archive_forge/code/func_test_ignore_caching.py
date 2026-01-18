from breezy import bedding, ignores, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_ignore_caching(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['ignoreme'])
    self.assertEqual(None, tree.is_ignored('ignoreme'))
    tree.unknowns()
    self.build_tree_contents([('.bzrignore', b'ignoreme')])
    self.assertEqual('ignoreme', tree.is_ignored('ignoreme'))