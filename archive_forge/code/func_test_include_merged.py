import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_include_merged(self):
    expected = ['2', '1.1.2', '1.2.1', '1.1.1', '1']
    self.assertLogRevnos(['--include-merged'], expected, working_dir='level0')
    self.assertLogRevnos(['--include-merged'], expected, working_dir='level0')