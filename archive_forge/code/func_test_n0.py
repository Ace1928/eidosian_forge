import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_n0(self):
    self.assertLogRevnos(['-n0', '-r1.1.1..1.1.4'], ['1.1.4', '4', '1.1.3', '1.1.2', '3', '1.1.1'])