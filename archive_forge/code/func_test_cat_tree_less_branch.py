from breezy.tests import TestCaseWithTransport
def test_cat_tree_less_branch(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('This revision', rev_id=b'abcd')
    tree.controldir.destroy_workingtree()
    output, errors = self.run_bzr(['cat-revision', '-d', 'a', 'abcd'])
    self.assertContainsRe(output, 'This revision')
    self.assertEqual('', errors)