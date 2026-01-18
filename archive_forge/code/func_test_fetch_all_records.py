import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_fetch_all_records(self):
    self.make_branch_and_tree('source')
    self.make_branch_and_tree('dest')
    out, err = self.run_bzr('fetch-all-records source -d dest')
    self.assertEqual(out, 'Done.\n')
    self.assertEqual(err, '')