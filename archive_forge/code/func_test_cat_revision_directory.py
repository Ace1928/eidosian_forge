from breezy.tests import TestCaseWithTransport
def test_cat_revision_directory(self):
    """Test --directory option"""
    tree = self.make_branch_and_tree('a')
    tree.commit('This revision', rev_id=b'abcd')
    output, errors = self.run_bzr(['cat-revision', '-d', 'a', 'abcd'])
    self.assertContainsRe(output, 'This revision')
    self.assertEqual('', errors)