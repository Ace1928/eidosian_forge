import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_merges_partial_range_ignore_before_lower_bound(self):
    """Dont show revisions before the lower bound's merged revs"""
    self.assertLogRevnosAndDepths(['-n0', '-r1.1.2..2'], [('2', 0), ('1.1.2', 1), ('1.2.1', 2)], working_dir='level0')