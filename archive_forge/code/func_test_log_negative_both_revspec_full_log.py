import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_negative_both_revspec_full_log(self):
    self.make_linear_branch()
    self.assertLogRevnos(['-r-3..-1'], ['3', '2', '1'])