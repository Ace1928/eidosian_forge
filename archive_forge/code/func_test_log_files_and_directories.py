import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_files_and_directories(self):
    """Logging files and directories together should be fine."""
    self.prepare_tree()
    self.assertLogRevnos(['file4', 'dir1/dir2'], ['4', '3'])