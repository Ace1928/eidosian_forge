from breezy.tests import TestCaseWithTransport
def test_already_url(self):
    wt = self.make_branch_and_tree('.')
    out, err = self.run_bzr('resolve-location %s' % wt.branch.user_url)
    self.assertEqual(out, '%s\n' % wt.branch.user_url.replace('file://', ''))