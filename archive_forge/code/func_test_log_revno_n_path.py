import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_revno_n_path(self):
    self.make_linear_branch('branch2')
    self.assertLogRevnos(['-rrevno:1:branch2'], ['1'])
    rev_props = self.log_catcher.revisions[0].rev.properties
    self.assertEqual('branch2', rev_props['branch-nick'])