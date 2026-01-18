import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_file_historical_missing(self):
    self.prepare_tree(complex=True)
    err_msg = 'Path unknown at end or start of revision range: file2'
    err = self.run_bzr('log file2', retcode=3)[1]
    self.assertContainsRe(err, err_msg)