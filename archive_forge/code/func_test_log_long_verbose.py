import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_long_verbose(self):
    self.assertUseLongDeltaFormat(['log', '--long', '-v'])