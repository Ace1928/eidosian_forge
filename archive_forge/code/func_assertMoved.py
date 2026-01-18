import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def assertMoved(self, from_path, to_path):
    """Assert that to_path is existing and versioned but from_path not. """
    self.assertPathDoesNotExist(from_path)
    self.assertNotInWorkingTree(from_path)
    self.assertPathExists(to_path)
    self.assertInWorkingTree(to_path)