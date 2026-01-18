import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_exclude_ancestry_single_revision(self):
    self.make_merged_branch()
    self.run_bzr_error(['brz: ERROR: --exclude-common-ancestry requires two different revisions'], ['log', '--exclude-common-ancestry', '-r1.1.1..1.1.1'])