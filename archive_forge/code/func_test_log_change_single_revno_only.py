import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_change_single_revno_only(self):
    self.make_minimal_branch()
    self.run_bzr_error(['brz: ERROR: Option --change does not accept revision ranges'], ['log', '--change', '2..3'])