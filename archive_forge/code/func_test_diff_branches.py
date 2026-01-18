import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_branches(self):
    self.example_branches()
    self.check_b2_vs_b1('diff -r branch:branch2 branch1')
    self.check_b2_vs_b1('diff --old branch2 --new branch1')
    self.check_b2_vs_b1('diff --old branch2 branch1')
    self.check_b2_vs_b1('diff branch2 --new branch1')
    self.check_b2_vs_b1('diff --old branch2 --new branch1 file')
    self.check_b2_vs_b1('diff --old branch2 branch1/file')
    self.check_b2_vs_b1('diff branch2/file --new branch1')
    self.check_no_diffs('diff --old branch2 --new branch1 file2')
    self.check_no_diffs('diff --old branch2 branch1/file2')
    self.check_no_diffs('diff branch2/file2 --new branch1')