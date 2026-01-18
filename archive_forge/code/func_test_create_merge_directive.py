from breezy import merge_directive
from breezy.bzr import chk_map
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_create_merge_directive(self):
    source_branch, target_branch = self.make_two_branches()
    directive = self.create_merge_directive(source_branch, target_branch.base)
    self.assertIsInstance(directive, merge_directive.MergeDirective2)