import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_range_open_begin(self):
    stdout, stderr = self.run_bzr(['log', '-r..2'], retcode=3)
    self.assertEqual(['2', '1'], [r.revno for r in self.get_captured_revisions()])
    self.assertEqual('brz: ERROR: Further revision history missing.\n', stderr)