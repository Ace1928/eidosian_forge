import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def check_unknowns(expected):
    self.assertEqual(expected, list(self.wt.unknowns()))