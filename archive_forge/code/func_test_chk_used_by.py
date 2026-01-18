import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_chk_used_by(self):
    self.make_branch_and_tree('.')
    out, err = self.run_bzr('chk-used-by chk')
    self.assertEqual(out, '')
    self.assertEqual(err, '')