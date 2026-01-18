import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_file1(self):
    self.prepare_tree()
    self.assertLogRevnos(['-n0', 'file1'], ['1'])