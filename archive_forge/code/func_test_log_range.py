import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_range(self):
    self.assertLogRevnos(['-r1..2'], ['2', '1'])