import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_change_incompatible_with_revision(self):
    self.run_bzr_error(['brz: ERROR: --revision and --change are mutually exclusive'], ['log', '--change', '2', '--revision', '3'])