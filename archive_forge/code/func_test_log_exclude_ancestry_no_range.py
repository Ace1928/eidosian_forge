import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_exclude_ancestry_no_range(self):
    self.make_linear_branch()
    self.run_bzr_error(['brz: ERROR: --exclude-common-ancestry requires -r with two revisions'], ['log', '--exclude-common-ancestry'])