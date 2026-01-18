import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_change_nonexistent_revno(self):
    self.make_minimal_branch()
    self.run_bzr_error(["brz: ERROR: Requested revision: '1234' does not exist in branch:"], ['log', '-c1234'])