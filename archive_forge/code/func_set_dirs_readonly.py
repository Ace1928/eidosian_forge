import os
import sys
import time
from breezy import tests
from breezy.bzr import hashcache
from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def set_dirs_readonly(self, basedir):
    """Set all directories readonly, and have it cleanup on test exit."""
    self.addCleanup(self._set_all_dirs, basedir, readonly=False)
    self._set_all_dirs(basedir, readonly=True)