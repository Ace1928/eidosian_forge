import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_reversed_revspecs(self):
    self.make_linear_branch()
    self.run_bzr_error(('brz: ERROR: Start revision must be older than the end revision.\n',), ['log', '-r3..1'])