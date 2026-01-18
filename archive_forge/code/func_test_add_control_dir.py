import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_control_dir(self):
    """The control dir and its content should be refused."""
    self.make_branch_and_tree('.')
    err = self.run_bzr('add .bzr', retcode=3)[1]
    self.assertContainsRe(err, 'ERROR:.*\\.bzr.*control file')
    err = self.run_bzr('add .bzr/README', retcode=3)[1]
    self.assertContainsRe(err, 'ERROR:.*\\.bzr.*control file')
    self.build_tree(['.bzr/crescent'])
    err = self.run_bzr('add .bzr/crescent', retcode=3)[1]
    self.assertContainsRe(err, 'ERROR:.*\\.bzr.*control file')