import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def run_bzr_remove_changed_files(self, files_to_remove, working_dir=None):
    self.run_bzr(['remove'] + list(files_to_remove), working_dir=working_dir)