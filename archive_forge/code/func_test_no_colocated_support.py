from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport
def test_no_colocated_support(self):
    self.run_bzr('init a')
    out, err = self.run_bzr('branches a')
    self.assertEqual(out, '* (default)\n')