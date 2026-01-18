from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_missing_repository(self):
    out, err = self.run_bzr('check --repo %s' % self.get_readonly_url(''))
    self.assertEqual(err, 'No repository found at specified location.\n')