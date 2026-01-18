from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def assertRemovedAndNotDeleted(self, files):
    self.assertNotInWorkingTree(files)
    self.assertPathExists(files)