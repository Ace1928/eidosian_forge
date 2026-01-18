import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_unversioned_non_ascii(self):
    """Clear error on mv of an unversioned non-ascii file, see lp:707954"""
    self.requireFeature(UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['§'])
    out, err = self.run_bzr_error(['Could not rename', 'not versioned'], ['mv', '§', 'b'])