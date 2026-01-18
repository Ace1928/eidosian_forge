import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_revno_n_path_wrong_namespace(self):
    self.make_linear_branch('branch1')
    self.make_linear_branch('branch2')
    self.run_bzr('log -r revno:2:branch1..revno:3:branch2', retcode=3)