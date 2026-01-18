import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_revno_n_path_correct_order(self):
    self.make_linear_branch('branch2')
    self.assertLogRevnos(['-rrevno:1:branch2..revno:3:branch2'], ['3', '2', '1'])