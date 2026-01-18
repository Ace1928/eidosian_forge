from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_missing_branch(self):
    out, err = self.run_bzr('check --branch %s' % self.get_readonly_url(''))
    self.assertEqual(err, 'No branch found at specified location.\n')