from breezy import bedding, ignores, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_mixed_is_ignored(self):
    tree = self.make_branch_and_tree('.')
    ignores._set_user_ignores(['*.py[co]', './.shelf'])
    self.build_tree_contents([('.bzrignore', b'./rootdir\n*.swp\n')])
    self.assertEqual('*.py[co]', tree.is_ignored('foo.pyc'))
    self.assertEqual('./.shelf', tree.is_ignored('.shelf'))
    self.assertEqual('./rootdir', tree.is_ignored('rootdir'))
    self.assertEqual('*.swp', tree.is_ignored('foo.py.swp'))
    self.assertEqual('*.swp', tree.is_ignored('.foo.py.swp'))
    self.assertEqual(None, tree.is_ignored('.foo.py.swo'))