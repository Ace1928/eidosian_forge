import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_file_historical_start(self):
    self.prepare_tree(complex=True)
    self.assertLogRevnos(['file1'], [])