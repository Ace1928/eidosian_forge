import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_dry_run(self):
    """Test a dry run add, make sure nothing is added."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['inertiatic/', 'inertiatic/esp'])
    self.assertEqual(list(wt.unknowns()), ['inertiatic'])
    self.run_bzr('add --dry-run')
    self.assertEqual(list(wt.unknowns()), ['inertiatic'])