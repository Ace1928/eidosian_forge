import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_change_revno(self):
    self.make_linear_branch()
    self.assertLogRevnos(['-c1'], ['1'])