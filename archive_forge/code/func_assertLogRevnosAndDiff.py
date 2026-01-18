import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def assertLogRevnosAndDiff(self, args, expected, working_dir='.'):
    self.run_bzr(['log', '-p'] + args, working_dir=working_dir)
    expected_revnos_and_depths = [(revno, depth) for revno, depth, diff in expected]
    self.assertEqual(expected_revnos_and_depths, [(r.revno, r.merge_depth) for r in self.get_captured_revisions()])
    fmt = 'In revno %s\n%s'
    for expected_rev, actual_rev in zip(expected, self.get_captured_revisions()):
        revno, depth, expected_diff = expected_rev
        actual_diff = actual_rev.diff
        self.assertEqualDiff(fmt % (revno, expected_diff), fmt % (revno, actual_diff))