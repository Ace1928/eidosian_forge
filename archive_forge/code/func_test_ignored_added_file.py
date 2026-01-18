from breezy.tests import TestCaseWithTransport
def test_ignored_added_file(self):
    """'brz ignored' should not list versioned files."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo.pyc'])
    self.build_tree_contents([('.bzrignore', b'foo.pyc')])
    self.assertTrue(tree.is_ignored('foo.pyc'))
    tree.add('foo.pyc')
    out, err = self.run_bzr('ignored')
    self.assertEqual('', out)
    self.assertEqual('', err)