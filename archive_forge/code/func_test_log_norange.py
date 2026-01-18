import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_norange(self):
    self.assertLogRevnos([], ['2', '1'])