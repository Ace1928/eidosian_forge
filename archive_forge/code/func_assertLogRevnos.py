import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def assertLogRevnos(self, args, expected_revnos, working_dir='.', out='', err=''):
    actual_out, actual_err = self.run_bzr(['log'] + args, working_dir=working_dir)
    self.assertEqual(out, actual_out)
    self.assertEqual(err, actual_err)
    self.assertEqual(expected_revnos, [r.revno for r in self.get_captured_revisions()])