import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_file_historical_end(self):
    self.prepare_tree(complex=True)
    self.assertLogRevnos(['-n0', '-r..4', 'file2'], ['4', '3.1.1', '2'])