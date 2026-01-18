import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testFirstLineRenumber(self):
    """Make sure we handle lines at the beginning of the hunk"""
    patch = parse_patch(self.datafile('insert_top.patch'))
    self.assertEqual(patch.pos_in_mod(0), 1)