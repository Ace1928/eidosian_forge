import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def assertFilesUnversioned(self, files):
    for f in files:
        self.assertNotInWorkingTree(f)
        self.assertPathExists(f)