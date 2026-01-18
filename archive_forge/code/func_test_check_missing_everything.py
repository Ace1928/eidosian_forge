from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_missing_everything(self):
    out, err = self.run_bzr('check %s' % self.get_readonly_url(''))
    self.assertEqual(err, 'No working tree found at specified location.\nNo branch found at specified location.\nNo repository found at specified location.\n')