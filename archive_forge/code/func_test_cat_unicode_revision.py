from breezy.tests import TestCaseWithTransport
def test_cat_unicode_revision(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('This revision', rev_id=b'abcd')
    output, errors = self.run_bzr(['cat-revision', 'abcd'])
    self.assertContainsRe(output, 'This revision')
    self.assertEqual('', errors)