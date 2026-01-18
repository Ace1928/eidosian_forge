from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_missing_tree(self):
    branch = self.make_branch('.')
    out, err = self.run_bzr('check --tree')
    self.assertEqual(err, 'No working tree found at specified location.\n')