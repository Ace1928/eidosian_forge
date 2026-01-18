import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_force_merge_revisions_on(self):
    self.assertLogRevnos(['-n0'], ['2', '1.1.2', '1.2.1', '1.1.1', '1'], working_dir='level0')