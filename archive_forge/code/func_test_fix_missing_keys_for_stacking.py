import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_fix_missing_keys_for_stacking(self):
    self.make_branch_and_tree('stacked')
    self.run_bzr('branch --stacked stacked new')
    out, err = self.run_bzr('fix-missing-keys-for-stacking new')
    self.assertEqual(out, '')
    self.assertEqual(err, '')