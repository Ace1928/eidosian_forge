import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_n1(self):
    self.assertLogRevnos(['-n1', '-r1.1.1..1.1.4'], ['1.1.4', '1.1.3', '1.1.2', '1.1.1'])