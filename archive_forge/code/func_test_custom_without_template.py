import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_custom_without_template(self):
    wt = self.make_branch_and_tree('branch')
    out, err = self.run_bzr('version-info --custom', retcode=3)
    self.assertContainsRe(err, 'ERROR: No template specified\\.')