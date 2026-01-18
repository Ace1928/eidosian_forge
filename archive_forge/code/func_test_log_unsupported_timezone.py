import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_unsupported_timezone(self):
    self.make_linear_branch()
    self.run_bzr_error(['brz: ERROR: Unsupported timezone format "foo", options are "utc", "original", "local".'], ['log', '--timezone', 'foo'])