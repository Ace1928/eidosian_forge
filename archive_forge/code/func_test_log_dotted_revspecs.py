import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_dotted_revspecs(self):
    self.make_merged_branch()
    self.assertLogRevnos(['-n0', '-r1..1.1.1'], ['1.1.1', '1'])